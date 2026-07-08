# This file contains code derived from:
# - https://github.com/topwasu/doing-experiments-and-revising-rules/
# Original authors: Top Piriyakulkij
import logging
from collections import Counter
from create_programs_from_string import convert_string_to_dsl
from data.pieces2tensor import prolog_strings_to_tensor
from data.tensor2piece import tensor_to_prolog_strings
from generation.render import render_scene
import numpy as np
import random
from itertools import chain
import re
import pprint
from program import Program, strip_trailing_var0
from scipy.optimize import minimize, Bounds
from scipy.stats import norm

from prompts.zendo import *
from models.prompters import get_prompter
from utils import extract_dsl_from_hypothesis, get_example_from_path, parse_listed_output, list_to_str, tensor_to_zendo_structure, ZendoPiece, ZendoStructure
from .utils import systematic_resample, feedback_generator, resample_optimal
import ast
from experiments.run_experiment import canonicalize_program, gather_data, normalize_program_structure
from pathlib import Path
import torch

log = logging.getLogger(__name__)

def normalize_rule(rule):
    strip_trailing_var0(rule)
    norm_rule = normalize_program_structure(rule)
    canonical_rule = canonicalize_program(norm_rule)
    return str(canonical_rule)

class LLMScientistPlayer:
      """LLM-driven scientist using a particle filter over hypotheses.

      Example format: ``((tensor_or_None, bool_label), path)``.
      """

      def __init__(
        self,
        player_id,
        task_idx,
        dsl,
        cfg,
        zendo_config,
        min_examples=4,
        prompter=None,
        retries=10,
        use_dsl=True,
        use_paths=True,
        seed=1,
        images=True,
      ):
        self.id = player_id
        self.task_idx = task_idx
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.config = {"proposal": "particle_filter",
                       "num_hs": 10,
                        "num_xs": 10,
                        "theta_mean": 0.7,
                        "theta_std": 0.1,
                        "delta_mean": 0.9,
                        "delta_std": 0.01,
                        "n_neighbors": 5,
                        "seed": seed}
        self.dsl = dsl
        self.cfg = cfg
        self.zendo_config = zendo_config
        self.proposal_config = {"name": "particle_filter",
            "num_hypotheses_per_call": 5,
            "num_max_calls_per_it": 5,
            "num_required_hypotheses": 10,
            "num_particles": 25,
            "deterministic": False}

        self.first_propose = True

        self.prompter = prompter
        if self.prompter is None:
            self.prompter = get_prompter(
                "gpt-5-mini", "zendo", seed, reasoning=False, sampling=False,
            )

        self.particles = []
        self.particle_weights = []
        self.pf_checkpoint = 0
        self.ct = 0

        if self.proposal_config["deterministic"]:
            self.thetas = np.ones(1)
            self.theta_priors = np.ones(1)
            self.deltas = np.ones(1)
            self.delta_priors = np.ones(1)
        else:
            self.thetas = np.arange(0.5, 1.01, 0.01)
            self.theta_priors = np.exp(np.asarray([norm.logpdf(theta, self.config["theta_mean"], self.config["theta_std"])
                                                   for theta in self.thetas]))
            self.deltas = np.arange(0.5, 1.01, 0.01)
            self.delta_priors = np.exp(np.asarray([norm.logpdf(delta, self.config["delta_mean"], self.config["delta_std"])
                                                   for delta in self.deltas]))
        self.all_priors = np.tile(self.theta_priors, len(self.deltas)) * np.repeat(self.delta_priors, len(self.thetas))

        self.examples = []
        self.program_cache = {}

        self.use_paths = use_paths
        self.guessing_stones = 0
        self.incorrect_rules = []
        self.last_label = None
        self.previous_guesses = []
        self.min_examples = min_examples
        self.cache_h = {}
        self.retries = retries
        self._pf_dirty = True
        self._pf_last_len = 0
        self._pf_support_cache = None
        self.use_dsl = use_dsl
        self.label_cache = {}
        self.already_guessed = []
        self.last_guessed_nl = None
        self.create_images = images

      def wrong_rule(self, rule):
        if rule not in self.incorrect_rules:
            self.incorrect_rules.append(normalize_rule(rule))
        self.top_guess = None

      def decide_guess(self, state):
        if self.guessing_stones <= 0 or len(self.examples) < self.min_examples:
            return None
        rule = self.guess_rule()
        if rule is None:
            log.info("Player %s could not find a rule", self.id)
            return None
        self.guessing_stones -= 1
        log.info("Player %s guessed rule: %s", self.id, rule)
        return {"type": "guess_rule", "rule": rule}
      
      def observe(self, example) -> None:
        (x, y), path = example
        if x is None:
            for _ in range(self.retries):
                x = get_example_from_path(path, self.prompter, seed=self.config["seed"])
                if x is not None:
                    break
        yb = bool(y)
        self.examples.append(((x, yb), path))
        self.last_label = yb

        # Original PF expects exactly one update per new datum.
        self.pf_checkpoint = len(self.examples)
        self._pf_dirty = True

      # ── Particle filter helpers ───────────────────────────────────────────

      def _get_c(self):
        class _C:
            def __init__(self, exs):
                self.paths = [p for ((_x, _y), p) in exs]
                self.labels = ['yes' if y else 'no' for ((_x, y), _p) in exs]
                self.structures = [x for ((x, _y), _p) in exs]
            def __len__(self):
                return len(self.structures)
        return _C(self.examples)

      def a_update_pf_if_needed(self):
            c = self._get_c()
            if not self._pf_dirty and self._pf_last_len == len(c) and self._pf_support_cache is not None:
                return

            if len(c) == 0:
                self._pf_support_cache = ["I don't know"]
                self._pf_last_len = 0
                self._pf_dirty = False
                return

            # IMPORTANT: only run PF once per dataset length
            self.a_particle_filter(c)

            # _pf_support_cache is set inside a_particle_filter from the unique hypothesis dict
            if self._pf_support_cache is None:
                self._pf_support_cache = ["I don't know"]
            self._pf_last_len = len(c)
            self._pf_dirty = False
      # ── NL hypothesis -> program ──────────────────────────────────────────

      def a_h2prog(self, h):
        if h == "I don't know":
            return None
        if h in self.program_cache:
            return self.program_cache[h]

        if self.use_dsl:
            prog = extract_dsl_from_hypothesis(h, self.prompter, self.cfg, self.config["seed"])
        else:
            outputs = self.prompter.prompt_with_text(rule_translation_prompt.format(h=h), seed=self.config["seed"])
            prog = outputs

        self.program_cache[h] = prog
        return prog

      # ── Likelihood ────────────────────────────────────────────────────────

      def eval_y_given_xprog(self, y, x, x_path, prog, hypothesis, thetas, deltas):
        by = (y == 'yes')
        if prog is None and hypothesis == "I don't know":
            return np.full(len(thetas) * len(deltas), 0.50)
        
        if hypothesis not in self.label_cache:
            self.label_cache[hypothesis] = {}
        mode = "dsl" if self.use_dsl else "llm"
        cache_key = x_path if self.create_images else str(x)
        if cache_key in self.label_cache[hypothesis] and mode in self.label_cache[hypothesis][cache_key]:
            res = self.label_cache[hypothesis][cache_key][mode]
        else:
            if self.use_dsl:
                prog_fn = prog.eval(dsl=self.dsl, environment=(None, None), i=0)
                res = bool(prog_fn(x))
            else:
                zendo_structure = tensor_to_zendo_structure(x)
                log.info(f"Evaluating hypothesis '{hypothesis}' on example '{zendo_structure}'")
                try:
                    namespace = {
                        "ZendoStructure": ZendoStructure,
                        "ZendoPiece": ZendoPiece,
                    }
                    exec(prog, namespace)
                    res = namespace["rule"](zendo_structure)
                except Exception as e:
                    log.info(f"Error evaluating program for hypothesis '{hypothesis}' on example '{zendo_structure}': {e}")
                    res = False
            if cache_key not in self.label_cache[hypothesis]:
                self.label_cache[hypothesis][cache_key] = {}

            self.label_cache[hypothesis][cache_key][mode] = res

        thetas = np.asarray(thetas)
        if not res:
            if by == res:
                return np.repeat(deltas, len(thetas))
            else:
                return np.repeat(1. - deltas, len(thetas))
        else:
            if by:
                return np.tile(thetas, len(deltas))
            else:
                return np.tile(1. - thetas, len(deltas))

      def a_score_h(self, h):
        # Heuristic length prior: shorter hypotheses preferred.
        return 1 / max(1, len(h.split(' ')))

      def a_score_joints(self, hs, c, noprior=False):
        unique_hs = list(set(hs))
        outputs = [self.a_h2prog(h) for h in unique_hs]
        h2prog = {h: prog for h, prog in zip(unique_hs, outputs)}

        res = []
        for h in hs:
            args_list = [(y, x, x_path, h2prog[h], h, self.thetas, self.deltas) for x, y, x_path in zip(c.structures, c.labels, c.paths)]
            y_given_xh_res = np.asarray([self.eval_y_given_xprog(*args) for args in args_list])
            sm = np.sum(np.prod(y_given_xh_res, axis=0) * self.all_priors)
            if noprior:
                res.append(sm)
            else:
                prior = self.a_score_h(h)
                res.append(prior * sm)
        return res

      # ── Particle filter ───────────────────────────────────────────────────

      def deduplicate_particles(self, particles):
        return [p.split('##')[-1] for p in particles]

      def duplicate_particles(self, particles, num, single=False):
        if single:
            x = particles
            return [f'{id}##' + x for id in range(num)]
        return sum([[f'{id}##' + x for id in range(num)] for x in particles], [])

      def a_sample_potential_hs(self, x_path: str, x_tensor):
        """Sample candidate hypotheses from the LLM, one prompt per attribute, returning a flat list."""
        prompts = []
        if self.use_paths:
            for att in self.zendo_config.att:
                prompt_text = basic_propose_h_prompt.format(
                    att=att,
                    att_choices=self.zendo_config.att_choices[att],
                    example=self.zendo_config.examples_texts_per_att[att],
                    num=self.proposal_config["num_hypotheses_per_call"],
                )
                prompts.append((prompt_text, [str(x_path)]))
        else:
            structure = tensor_to_prolog_strings([x_tensor])[0]
            for att in self.zendo_config.att:
                prompt_text = basic_propose_h_prompt_desc.format(
                    att=att,
                    att_choices=self.zendo_config.att_choices[att],
                    example=self.zendo_config.examples_texts_per_att_desc[att],
                    num=self.proposal_config["num_hypotheses_per_call"],
                    x=structure,
                )
                prompts.append((prompt_text, []))

        outputs = []
        for prompt_text, paths in prompts:
            if paths:
                outputs.append(self.prompter.prompt_with_images(
                    prompt_text=prompt_text, paths=paths, seed=self.config["seed"],
                ))
            else:
                outputs.append(self.prompter.prompt_with_text(
                    prompt_text=prompt_text, seed=self.config["seed"],
                ))

        potential_hs = np.concatenate([parse_listed_output(output) for output in outputs])
        potential_hs = self.np_rng.permutation(potential_hs)
        return potential_hs
      
      def quiz_correct(self):
        self.guessing_stones += 1
        
      def quiz_incorrect(self):
        self.top_guess = None

      def a_get_rejuvenation_options(self, h, x_path, x_tensor, y, c):
        if self.use_paths:
            prompt_text = new_evolve_h_prompt.format(
                h=h,
                text_y='' if y == 'yes' else 'NOT',
                num=self.proposal_config["num_hypotheses_per_call"],
            )
            paths = [str(x_path)]
        else:
            structure = tensor_to_prolog_strings([x_tensor])[0]
            prompt_text = new_evolve_h_prompt_desc.format(
                h=h,
                text_y='' if y == 'yes' else 'NOT',
                num=self.proposal_config["num_hypotheses_per_call"],
                x=structure,
            )
            paths = []

        try:
            if paths:
                response = self.prompter.prompt_with_images(
                    prompt_text=prompt_text, paths=paths, seed=self.config["seed"],
                )
            else:
                response = self.prompter.prompt_with_text(
                    prompt_text=prompt_text, seed=self.config["seed"],
                )
        except Exception as e:
            log.info("Error during LLM call for rejuvenation options: %s", e)
            return [], []

        outputs = [response]
        rules = sum([parse_listed_output(output) for output in outputs], [])
        rules = np.asarray([rule.split('->')[-1].strip(" '\n") for rule in rules])
        probs = self.a_score_joints(rules, c)
        probs = np.asarray(probs, dtype=np.float64)
        return rules, probs

      def a_particle_filter(self, c):
        """
        Mostly the same as your snippet, but:
          - no ZendoGame objects
          - need tensor->text conversion for prompts in initialization/rejuvenation
        """
        if len(c) != self.pf_checkpoint:
            raise Exception('Particle filter called out of sync with observations.')

        old_particles, old_particle_weights = self.particles, self.particle_weights

        if self.first_propose:
            # Seed PF from a random observed positive (fall back to most recent example).
            pos = [x for x, y in zip(c.paths, c.labels) if y == 'yes']
            pos_tensors = [x for x, y in zip(c.structures, c.labels) if y == 'yes']
            if len(pos) == 0:
                seed_x = c.paths[-1]
                seed_x_tensor = c.structures[-1]
            else:
                idx = self.rng.randrange(len(pos))
                seed_x = pos[idx]
                seed_x_tensor = pos_tensors[idx]

            potential_hs = self.a_sample_potential_hs(seed_x, seed_x_tensor)
            log.info(f'Initial potential hypotheses: {potential_hs}')
            probs = self.a_score_joints(potential_hs, c)
            probs = np.asarray(probs, dtype=np.float64)

            self.particles = np.asarray(potential_hs, dtype='object')
            self.particle_weights = np.asarray(probs)
            if np.sum(self.particle_weights) == 0:
                self.particle_weights[:] = 1.0
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

        else:
            to_evolve = {}

            log.info('Rejuvenation...')

            # Score unique particles to avoid redundant LLM calls on duplicates.
            unique_particles = list(dict.fromkeys(self.particles))
            tplusone_likelihoods_unique = self.a_score_joints(unique_particles, c, noprior=True)
            unique_score = {h: s for h, s in zip(unique_particles, tplusone_likelihoods_unique)}
            tplusone_likelihoods = [unique_score[p] for p in self.particles]
            sorted_indices = np.argsort(tplusone_likelihoods)

            for idx in sorted_indices:
                if self.proposal_config["num_max_calls_per_it"] == len(to_evolve) and self.particles[idx] not in to_evolve:
                    continue
                if tplusone_likelihoods[idx] == 1:  # Don't rejuvenate already-perfect rules.
                    continue
                to_evolve[self.particles[idx]] = True
            to_evolve = list(to_evolve.keys())

            x_last = c.paths[-1]
            x_last_tensor = c.structures[-1]
            y_last = c.labels[-1]

            evolve_options_list = [
                self.a_get_rejuvenation_options(h, x_last, x_last_tensor, y_last, c)
                for h in to_evolve
            ]
            evolve_options_dict = dict(zip(to_evolve, evolve_options_list))

            new_particles = []
            probs = self.a_score_joints(to_evolve, c)
            for (k, (options, option_scores)), p in zip(evolve_options_dict.items(), probs):
                options, option_scores = np.asarray(options), np.asarray(option_scores)
                options, option_scores = options[option_scores > p], option_scores[option_scores > p]

                if len(option_scores) > 0:
                    options_prob = np.asarray(option_scores) / sum(option_scores)
                else:
                    options_prob = np.asarray([])

                if len(options_prob) > self.config["n_neighbors"]:
                    indices = systematic_resample(options_prob, self.config["n_neighbors"], rng=self.np_rng)
                    new_particles.append(options[indices])
                    log.info(f"Rejuvenate '{k}' to '{options[indices]}'")
                elif len(options_prob) > 0:
                    new_particles.append(options)
                    log.info(f"Rejuvenate '{k}' to '{options}'")
                else:
                    log.info(f"Rejuvenate '{k}' to 'NOTHING'")
                new_particles.append([k])

            if len(to_evolve) == 0:
                new_particles = [self.particles]
            else:
                avg_upsample = 1
                new_particles = np.unique(np.concatenate(new_particles))
                new_particles = [self.duplicate_particles(new_particles, 1)]
                for p in self.particles:
                    if p not in to_evolve:
                        new_particles = new_particles + [self.duplicate_particles(p, avg_upsample, single=True)]

            self.particles = np.unique(np.concatenate(new_particles))
            self.particle_weights = np.ones(len(self.particles)) / len(self.particles)

            log.info('Reweighting...')
            new_joints = self.a_score_joints(self.deduplicate_particles(self.particles), c)

            for idx, new_joint in enumerate(new_joints):
                self.particle_weights[idx] *= new_joint

            if np.sum(self.particle_weights) == 0:
                self.particle_weights[:] = 1
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

        self.first_propose = False

        log.info('Resampling...')
        resampled_indices = systematic_resample(self.particle_weights, self.proposal_config["num_particles"], rng=self.np_rng)
        log.info('Resampled indices: %s, particles: %s', resampled_indices, self.particles)
        self.particles = self.particles[resampled_indices]
        self.particle_weights = np.ones(len(self.particles)) / len(self.particles)
        self.particles = self.deduplicate_particles(self.particles)

        # Build unique hypothesis dict (matching reference): iterate zip so duplicates collapse
        hs_dict = {}
        for h, p in zip(self.particles, self.particle_weights):
            if p > 0:
                hs_dict[h] = p

        old_hs_dict = {h: True for h in old_particles} if len(old_particles) else {}
        log.info('Changes to particles:')
        for h in old_hs_dict:
            if h not in hs_dict:
                log.info('DELETED %s', h)
        for h in hs_dict:
            if h not in old_hs_dict:
                log.info('ADDED %s', h)

        res = self.np_rng.permutation(list(hs_dict))
        self._pf_support_cache = list(res)
        log.info('Current working rules\n%s', pprint.pformat(list(res)))
        return res

      # ── Prediction / label guessing ───────────────────────────────────────

      def a_dist_y_given_cx(self, c, x, x_path):
        hs = self.a_sample_proposal_q(c)
        dist_y_given_cx = []
        for y in ['yes', 'no']:
            class _Cxy:
                def __init__(self, c, x, y, x_path):
                    self.structures = list(c.structures) + [x]
                    self.labels = list(c.labels) + [y]
                    self.paths = list(c.paths) + [x_path]
                def __len__(self):
                    return len(self.structures)

            cxy = _Cxy(c, x, y, x_path)
            joints = self.a_score_joints(hs, cxy)
            dist_y_given_cx.append(np.sum(joints))

        if np.sum(dist_y_given_cx) == 0:
            dist_y_given_cx = np.ones_like(dist_y_given_cx)
        dist_y_given_cx = np.asarray(dist_y_given_cx) / np.sum(dist_y_given_cx)
        return dist_y_given_cx

      def a_sample_proposal_q(self, c):
            self.a_update_pf_if_needed()
            return self._pf_support_cache[: self.config["num_hs"]]

      def a_get_query_x(self, c):
          log.info('Getting query x...')
          xs = self.a_get_relevant_xs(c)
          expected_kl_divergences = [self.a_get_expected_kl_divergence(c, x[0], x[1]) for x in xs]
          best_idx = np.argmax(expected_kl_divergences)
          if expected_kl_divergences[best_idx] == 0:
            log.info('NOTE Expected kl divergence is 0, trying random queries now')
            best_idx = self.np_rng.choice(len(xs))
          return xs[best_idx]

      def a_get_expected_kl_divergence(self, c, x, x_path):
        dist_y_given_cx = self.a_dist_y_given_cx(c, x, x_path)

        if np.all(dist_y_given_cx == 0):
            return 0

        kl_divergences = [
            self.a_get_kl_divergence(c, x, 'yes', x_path),
            self.a_get_kl_divergence(c, x, 'no', x_path),
        ]
        
        if np.max(kl_divergences) == 1000000:
            return 1e-6
        
        return np.sum(dist_y_given_cx * np.asarray(kl_divergences))
      
      def a_dist_h_given_c_with_support(self, c, support):
        joints = self.a_score_joints(support, c)
        if np.sum(joints) == 0: # This is possible if support is totally inconsistent with cs
          return np.zeros_like(joints)
        return np.asarray(joints) / np.sum(joints)

      def a_get_kl_divergence(self, c, x, y, x_path):
        support = self.a_sample_proposal_q(c)
        class _Cxy:
                def __init__(self, c, x, y, x_path):
                    self.structures = list(c.structures) + [x]
                    self.labels = list(c.labels) + [y]
                    self.paths = list(c.paths) + [x_path]
                def __len__(self):
                    return len(self.structures)
        new_c = _Cxy(c, x, y, x_path)
        phat_h_given_cxy = self.a_dist_h_given_c_with_support(new_c, support)
        phat_h_given_c = self.a_dist_h_given_c_with_support(c, support)

        # If everything is equal to zero given the support, that means things are very different
        if np.all(phat_h_given_cxy == 0):
            return 1000000

        kl_divergence = 0
        for i in range(len(phat_h_given_cxy)):
            # Skip zero entries: 0 * log(0) is treated as 0 here.
            if phat_h_given_cxy[i] != 0:
                kl_divergence += phat_h_given_cxy[i] * (np.log(phat_h_given_cxy[i]) - np.log(phat_h_given_c[i]))
        return kl_divergence

      def a_get_relevant_xs(self, c):
        log.info('Getting relevant xs...')
        hs = self.a_sample_proposal_q(c)
        log.info(f'Coming up with xs...', hs)
        res = self.a_sample_proposal_r(hs)
        filtered_res = [x for x in res if len(x[0]) > 0]
        return filtered_res

      def a_sample_proposal_r(self, hs):
        target = self.config["num_xs"]
        res = []

        for attempt in range(self.retries):
            log.info(f'Attempt {attempt} to generate novel structure...')

            all_xs_txt = [
                self.a_sample_proposal_r_info_gain(
                    h,
                    rng=np.random.default_rng(self.np_rng.integers(500)),
                )
                for h in hs
            ]
            all_xs_txt = list(chain.from_iterable(x for x in all_xs_txt if x is not None))
            self.rng.shuffle(all_xs_txt)

            log.info(f"Got {len(all_xs_txt)} candidates (need {target - len(res)} more)")

            for i, xs_txt in enumerate(all_xs_txt):
                if len(res) >= target:
                    break

                try:
                    structure = prolog_strings_to_tensor([xs_txt[0]])[0]
                except (KeyError, ValueError, IndexError) as e:
                    log.warning("Skipping invalid structure %s: %s", xs_txt[0], e)
                    continue
                if structure is None:
                    log.warning("Skipping None structure for %s", xs_txt[0])
                    continue
                log.info("Generated structure: %s", structure)
                if self.create_images:
                    candidate_path = Path("KL_divergence") /Path(str(self.task_idx)) / Path(str(self.id)) / str(len(self.examples)) / str(i)
                    full_input_path = Path("generation") / Path("output") / (str(candidate_path) + ".png")
                    new_input_rendered = render_scene(tensor_to_prolog_strings([structure])[0], path=candidate_path)

                    if new_input_rendered is not None:
                        res.append([new_input_rendered, full_input_path, xs_txt[1]])
                        log.info(f"Accepted ({len(res)}/{target})")
                else:
                    res.append([structure, "", xs_txt[1]])
                    log.info(f"Accepted ({len(res)}/{target})")

            if len(res) >= target:
                return res
        if res:
            log.info(f"Returning partial results: {len(res)}/{target}")
            return res

        log.info("All attempts failed.")
        return None

      def a_sample_proposal_r_info_gain(self, h, rng=None):
        if h in self.cache_h:
            return self.cache_h[h]

        cur_output = self.prompter.prompt_with_text(
            prompt_text=propose_x_prompt.format(h=h),
            seed=self.config["seed"],
        )
        xs = extract_labeled_structures(cur_output)

        if len(xs) != 2:
            log.info("Failed to extract exactly two labeled structures")
            return None
        self.cache_h[h] = xs
        return self.cache_h[h]

      # ── guess_label ───────────────────────────────────────────────────────

      def guess_label(self, input_scene) -> bool:
        if isinstance(input_scene, (tuple, list)) and len(input_scene) >= 1:
            x = input_scene[0]
        else:
            x = input_scene

        c = self._get_c()
        dist = self.a_dist_y_given_cx(c, x, input_scene[1] if isinstance(input_scene, (tuple, list)) and len(input_scene) >= 2 else "")
        return bool(dist[0] > dist[1])

      # ── guess_rule ────────────────────────────────────────────────────────

      def guess_rule(self) -> Program:
        c = self._get_c()
        if len(self.particles) == 0 and len(c) > 0:
            res = self.a_particle_filter(c)
            if res is not None and len(res) > 0:
                self._pf_support_cache = list(res)
            self._pf_last_len = len(c)
            self._pf_dirty = False

        if len(self.particles) == 0:
            return None

        # MAP estimate via duplicate counts left by systematic_resample.
        counts = Counter(self.particles)

        h = None
        for candidate, _ in counts.most_common():
            if candidate not in self.already_guessed and candidate != "I don't know":
                h = candidate
                break

        # Fallback: every candidate already guessed; pick the most probable anyway.
        if h is None:
            for candidate, _ in counts.most_common():
                if candidate != "I don't know":
                    h = candidate
                    break

        if h is None:
            return None

        self.already_guessed.append(h)

        if self.use_dsl:
            prog = self.a_h2prog(h)
            return prog
        else:
            return h

      def react(self, state):
        turn = state.current_turn.name
        if turn == "PROPOSE":
            return self._react_propose(state)
        elif turn == "LABEL":
            return self._react_label(state)
        elif turn == "GUESS":
            return self._react_guess(state)
        return None

      def _react_propose(self, state):
        proposed_input, rule, path = self.propose_input()
        amount_players = len(state.player_guess_tokens)
        if proposed_input is None:
            log.info("Failed to propose input, returning None.")
            return {"type": "propose_input", "input": None, "mode": "TELL"}
        mode = "QUIZ" if amount_players == 1 else "TELL"
        return {"type": "propose_input", "input": (proposed_input, path), "mode": mode, "rule": rule}

      def _react_label(self, state):
        label = self.guess_label(state.input_scene)
        return {"type": "guess_label", "label": label}

      def _react_guess(self, state):
        action = self.decide_guess(state)
        if action is None:
            return {"type": "no_guess"}
        return action

      def propose_input(self) -> list:
        """Use top PF hypotheses to query the LLM for a new structure; returns (tensor, rule_for_logging, path)."""
        c = self._get_c()

        hs = self.a_sample_proposal_q(c) if len(c) > 0 else ["I don't know"]

        for _ in range(self.retries):
            tensor, path, label = self.a_get_query_x(c)
            log.info("Proposed tensor: %s from path: %s", tensor, path)
            rule_str_for_logging = hs[0] if len(hs) else "I don't know"
            if tensor is not None and (not self.create_images or (path is not None and path != "")):
                return tensor, rule_str_for_logging, path
        return None, None, None

      def _handle_proposed_items(self, llm_answer: str) -> list:
        lines = [line for line in llm_answer.split("\n") if not line.strip().startswith("#")]
        response_text = "\n".join(lines).strip()
        if response_text.startswith("```python"):
            response_text = response_text.strip("`").split("python", 1)[-1].strip()
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0].strip()
        log.info("GPT-4o response: %s %s", response_text, type(response_text))
        if not isinstance(response_text, str):
            log.info("GPT-4o response is not a string.")
            return None
        try:
            parsed = ast.literal_eval(response_text)
        except Exception as e:
            log.info("Failed to parse response using ast.literal_eval: %s", e)
            return None

        if not isinstance(parsed, list) or len(parsed) != 2:
            log.info("Unexpected format. Expected: [[item strings...], label]")
            return None

        items = parsed[0]
        if not isinstance(items, list) or not all(isinstance(s, str) for s in items):
            log.info("Invalid item list.")
            return None
        input_tensor = prolog_strings_to_tensor([items])[0]
        log.info("llm response: %s Parsed tensor: %s", response_text, input_tensor)
        if input_tensor is None:
            log.info("Failed to parse llm response into tensor.")
            return None
        else:
            log.info("llm response successfully parsed into tensor.")
            return input_tensor
        

def extract_labeled_structures(text):
    """Extract all occurrences of ``[[item strings...], 0|1]`` from ``text``."""
    results = []

    pattern = re.compile(
        r'\[\s*\[\s*(?:".*?"\s*,?\s*)*\]\s*,\s*[01]\s*\]',
        re.DOTALL
    )

    matches = pattern.findall(text)

    for match in matches:
        try:
            parsed = ast.literal_eval(match)
        except Exception:
            continue

        if (
            isinstance(parsed, list)
            and len(parsed) == 2
            and isinstance(parsed[0], list)
            and all(isinstance(s, str) for s in parsed[0])
            and parsed[1] in (0, 1)
        ):
            results.append((parsed[0], parsed[1]))

    return results