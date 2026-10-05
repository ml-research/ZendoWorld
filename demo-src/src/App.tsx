import { useMemo, useState } from "react";
import type { SceneJSON } from "./types";
import SceneBuilder from "./builder/SceneBuilder";
import { POSITIVE_SCENES, NEGATIVE_SCENES } from "./seedScenes";
import { ruleMatches, trueLabel, TRUE_RULE_TEXT } from "./ruleCheck";
import "./App.css";

type Phase = "intro" | "build" | "label" | "rule" | "result";

export default function App() {
  const [phase, setPhase] = useState<Phase>("intro");
  const [scene, setScene] = useState<SceneJSON>(() => ({
    id: crypto.randomUUID(),
    size: 320,
    pieces: [],
  }));
  const [guessedLabel, setGuessedLabel] = useState<"YES" | "NO" | null>(null);
  const [ruleGuess, setRuleGuess] = useState("");
  const [outcome, setOutcome] = useState<"won" | "lost" | null>(null);

  const actualLabel = useMemo(() => trueLabel(scene), [scene]);
  const labelWasCorrect = guessedLabel !== null && guessedLabel === actualLabel;

  function reset() {
    setScene({ id: crypto.randomUUID(), size: 320, pieces: [] });
    setGuessedLabel(null);
    setRuleGuess("");
    setOutcome(null);
    setPhase("intro");
  }

  return (
    <div className="demo-root">
      <header className="demo-header">
        <h1>Try one turn of ZendoWorld</h1>
        <p className="demo-sub">
          A hidden rule labels every scene <strong>positive</strong> or{" "}
          <strong>negative</strong>. Study the six examples, build one new scene of
          your own, guess its label, then guess the rule.
        </p>
      </header>

      <section className="demo-seeds">
        <div className="demo-seeds-col">
          <h3>Positive examples</h3>
          <div className="demo-seeds-row">
            {POSITIVE_SCENES.map((src, i) => (
              <figure key={src} className="demo-seed demo-seed-yes">
                <img src={src} alt={`Positive example ${i + 1}`} />
                <figcaption>positive</figcaption>
              </figure>
            ))}
          </div>
        </div>
        <div className="demo-seeds-col">
          <h3>Negative examples</h3>
          <div className="demo-seeds-row">
            {NEGATIVE_SCENES.map((src, i) => (
              <figure key={src} className="demo-seed demo-seed-no">
                <img src={src} alt={`Negative example ${i + 1}`} />
                <figcaption>negative</figcaption>
              </figure>
            ))}
          </div>
        </div>
      </section>

      {phase === "intro" && (
        <section className="demo-card">
          <h2>Step 1 &mdash; propose a scene</h2>
          <p>
            Design a new scene you think will help you figure out the hidden rule.
            Drag pieces from the palette on the left; click a piece to cycle its
            orientation.
          </p>
          <button className="demo-btn primary" onClick={() => setPhase("build")}>
            Start building
          </button>
        </section>
      )}

      {(phase === "build" || phase === "label" || phase === "rule" || phase === "result") && (
        <section className="demo-builder-wrap">
          <SceneBuilder scene={scene} setScene={setScene} />
        </section>
      )}

      {phase === "build" && (
        <section className="demo-card">
          <h2>Step 2 &mdash; guess the label</h2>
          <p>
            Once your scene is ready, decide whether the hidden rule would label it
            positive or negative.
          </p>
          <button
            className="demo-btn primary"
            disabled={scene.pieces.length === 0}
            onClick={() => setPhase("label")}
            title={scene.pieces.length === 0 ? "Add at least one piece" : undefined}
          >
            Guess the label
          </button>
        </section>
      )}

      {phase === "label" && (
        <section className="demo-card">
          <h2>Is your scene positive or negative?</h2>
          <div className="demo-label-row">
            <button
              className="demo-btn label-yes"
              onClick={() => {
                setGuessedLabel("YES");
                setPhase("rule");
              }}
            >
              Positive
            </button>
            <button
              className="demo-btn label-no"
              onClick={() => {
                setGuessedLabel("NO");
                setPhase("rule");
              }}
            >
              Negative
            </button>
          </div>
        </section>
      )}

      {phase === "rule" && (
        <section className="demo-card">
          <div className={`demo-feedback ${labelWasCorrect ? "ok" : "bad"}`}>
            <strong>{labelWasCorrect ? "Correct label!" : "Wrong label."}</strong>{" "}
            The true label of your scene is <strong>{actualLabel === "YES" ? "positive" : "negative"}</strong>.
          </div>

          <h2>Step 3 &mdash; guess the rule</h2>
          <p>Type the rule in natural language. Example phrasings work too.</p>
          <input
            className="demo-input"
            type="text"
            value={ruleGuess}
            onChange={(e) => setRuleGuess(e.target.value)}
            placeholder="e.g. there are 4 red pieces"
            onKeyDown={(e) => {
              if (e.key === "Enter" && ruleGuess.trim()) {
                setOutcome(ruleMatches(ruleGuess) ? "won" : "lost");
                setPhase("result");
              }
            }}
          />
          <button
            className="demo-btn primary"
            disabled={!ruleGuess.trim()}
            onClick={() => {
              setOutcome(ruleMatches(ruleGuess) ? "won" : "lost");
              setPhase("result");
            }}
          >
            Submit rule
          </button>
        </section>
      )}

      {phase === "result" && (
        <section className={`demo-card demo-result ${outcome}`}>
          {outcome === "won" ? (
            <>
              <h2>You won!</h2>
              <p>You correctly identified the hidden rule.</p>
            </>
          ) : (
            <>
              <h2>You lost.</h2>
              <p>
                Your guess: <em>&ldquo;{ruleGuess}&rdquo;</em>
                <br />
                The true rule was: <strong>{TRUE_RULE_TEXT}</strong>
              </p>
            </>
          )}
          <button className="demo-btn" onClick={reset}>
            Play again
          </button>
        </section>
      )}

      <footer className="demo-footer">
        This is a simplified single-turn demo of the ZendoWorld human study. In the
        real study, a Prolog engine checks proposed rules and scenes are rendered
        with Blender; both are stripped out here so the demo can run entirely in
        your browser.
      </footer>
    </div>
  );
}
