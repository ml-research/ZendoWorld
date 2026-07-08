% This file contains code derived from:
% - https://github.com/CapArrow/zendo_game_dataset_generator

:- use_module(library(apply)).

%%% Facts %%%
orientation(flat).
orientation(upright).
orientation(upside_down).
orientation(cheesecake).
orientation(doorstop).
%orientation(weird).

color(blue).
color(yellow).
color(red).

shape(pyramid).
shape(wedge).
shape(block).

% Grounded = specific case: handled like attribute and not an interaction, so check with attribute rules !!!
interaction(grounded). % Never the only attribute in a rule, so always in combination with an orientation, color or shape attribute!
interaction(touching(_)).
interaction(pointing(_)).
interaction(on_top_of(_)).
%interaction(inside(_)).

max_items(5).
min_items(1).


%%% Generating %%%
% Generate repeatedly until a structure satisfies all checks (main function)
generate_valid_structure(Checks, Structure) :-
    determine_structure_size(Checks, N),
    repeat,
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    (and(Checks) -> !; fail).

% Generate repeatedly until a structure doesnt fulfills the checks
generate_invalid_structure(Checks, Structure) :-
    determine_invalid_structure_size(Checks, N),
    repeat,
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    (not(and(Checks)) -> !; fail).

% Generate a structure that disproves Checks1: invalid for Checks1, valid for Checks2
generate_disproving_structure(Checks1, Checks2, Structure) :-
    disproving_sizes(Checks1, Checks2, Sizes),
    member(N, Sizes),
    between(1, 100, _),
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    \+ and(Checks1),
    and(Checks2),
    !.

% Sizes to try when disproving Checks1 while satisfying Checks2. Must be a size
% that can satisfy Checks2 (hard constraint). Prefer sizes that also fail
% Checks1s size constraints (primary tier), falling back to other sizes valid
% for Checks2 where the disproof can come from non-size checks.
disproving_sizes(Checks1, Checks2, Sizes) :-
    ( possible_structure_sizes_from_checks(Checks2, S2) -> true ; default_size_range(S2) ),
    ( invalid_structure_sizes_from_checks(Checks1, Inv1) -> true ; default_size_range(Inv1) ),
    findall(N, (member(N, Inv1), member(N, S2)), Primary),
    findall(N, (member(N, S2), \+ member(N, Primary)), Secondary),
    random_permutation(Primary, PShuf),
    random_permutation(Secondary, SShuf),
    append(PShuf, SShuf, Sizes).

% Generate a structure that discriminates two rule sets (XOR: satisfies exactly one)
generate_discriminating_structure(Checks1, Checks2, Structure) :-
    discriminating_sizes(Checks1, Checks2, Sizes),
    random_permutation(Sizes, Shuffled),
    member(N, Shuffled),
    between(1, 100, _),
    generate_items(N, N, Structure),
    interaction_constraint_check(Structure),
    (
        (and(Checks1), \+ and(Checks2))
    ;
        (\+ and(Checks1), and(Checks2))
    ),
    !.

% Sizes that could possibly discriminate Checks1 from Checks2: sizes valid for
% exactly one of the two check sets. Falls back to the default range if both
% check sets place no size constraint (then any size might discriminate).
discriminating_sizes(Checks1, Checks2, Sizes) :-
    ( possible_structure_sizes_from_checks(Checks1, S1) -> true ; default_size_range(S1) ),
    ( possible_structure_sizes_from_checks(Checks2, S2) -> true ; default_size_range(S2) ),
    findall(N, (member(N, S1), \+ member(N, S2)), Only1),
    findall(N, (member(N, S2), \+ member(N, S1)), Only2),
    append(Only1, Only2, Combined),
    % Combined is empty only when S1 and S2 contain the same sizes; in that
    % case structures can still differ on non-size constraints, so try those
    % shared sizes (which equals the default range when neither check
    % constrains size).
    ( Combined \= [] -> Sizes = Combined ; Sizes = S1 ).

% Define preferred size range if not specified by rules
default_size_range([7, 6, 5, 4, 3, 2, 1]).

determine_structure_size(Checks, N) :-
    possible_structure_sizes_from_checks(Checks, ValidSizes),
    random_member(N, ValidSizes).

contains_explicitly_allows_1_in_either_or(Checks) :-
    member(C, Checks),
    (
        C = either_or(1, _, _)
    ;
        C = either_or(_, 1, _)
    ).

possible_structure_sizes_from_checks(Checks, ValidSizes) :-
    default_size_range(DefaultSizes),

    % 0. Exact length_ constraint overrides everything
    ( member(length_(FixedN, _), Checks) ->
        ValidSizes = [FixedN], !
    ; true ),

    % 1. Check for either_or constraints
    ( member(either_or(A, B, _), Checks) ->
        AllowedSizes1 = [A, B]
    ; AllowedSizes1 = DefaultSizes ),

    % 2. Check for odd/even number constraints
    ( member(odd_number_of(_), Checks) ->
        include(odd, AllowedSizes1, AllowedSizes2)
    ; member(even_number_of(_), Checks) ->
        include(even, AllowedSizes1, AllowedSizes2)
    ; AllowedSizes2 = AllowedSizes1 ),

    % 3. Find all minimums from any constraints that imply minimum structure size
    findall(N, (
        member(C, Checks),
        (
            C = exactly(_, N, _)
        ;   C = exactly(_, _, N, _)
        ;   C = at_least_interaction(_, _, _, N, _)
        )
    ), Ns),
    ( Ns = [] -> MinReq = 1 ; max_list(Ns, MinReq) ),

    % 4. Filter by minimum requirement
    include(ge(MinReq), AllowedSizes2, ValidSizes),
    ValidSizes \= [], !.  % Succeed only if non-empty

determine_invalid_structure_size(Checks, N) :-
    invalid_structure_sizes_from_checks(Checks, InvalidSizes),
    random_member(N, InvalidSizes).

invalid_structure_sizes_from_checks(Checks, Sizes) :-
    default_size_range(DefaultSizes),

    % Step 1: Exclude the fixed size from length_ constraint (invalid = wrong count)
    ( member(length_(FixedN, _), Checks) ->
        exclude(eq(FixedN), DefaultSizes, Sizes1)
    ; Sizes1 = DefaultSizes ),

    % Step 2: Exclude sizes listed in either_or
    ( member(either_or(EitherN, OrN, _), Checks) ->
        exclude(eq(EitherN), Sizes1, T2),
        exclude(eq(OrN), T2, Sizes2)
    ; Sizes2 = Sizes1 ),

    % Step 3: Apply opposite of odd/even
    ( member(odd_number_of(_), Checks) ->
        include(even, Sizes2, Sizes3)
    ; member(even_number_of(_), Checks) ->
        include(odd, Sizes2, Sizes3)
    ; Sizes3 = Sizes2 ),

    % Step 4: Filter out any structure sizes that exactly match attribute-based counts
    findall(N, (
        member(C, Checks),
        (C = exactly(_, N, _); C = exactly(_, _, N, _))
    ), ExactNs),
    exclude_list(Sizes3, ExactNs, Sizes),

    Sizes \= [], !.  % Fail safely if nothing matches


generate_structure(Structure) :-
    random_structure_size(N),
    repeat,
    (generate_items(N, N, Structure) -> !; fail).

generate_items(0, _, []) :- !.
generate_items(N, TotalN, [item(Id,C,S,O,I)|Rest]) :-
    Id is N - 1,
    random_color(C),
    random_shape(S),
    random_orientation(S, O),
    MaxId is TotalN - 1,
    random_interaction(MaxId, Id, I),
    N1 is N - 1,
    generate_items(N1, TotalN, Rest).

random_color(Color) :-
    findall(C, color(C), Colors),
    random_member(Color, Colors).

random_shape(Shape) :-
    findall(S, shape(S), Shapes),
    random_member(Shape, Shapes).

random_orientation(Shape, Orientation) :-
    %findall(O, orientation(O), Orientations),
    % exclude specific cases for the different shapes
    findall(O, (orientation(O), (
        (Shape = block, O \= cheesecake, O \= doorstop);
        (Shape = pyramid, O \= cheesecake, O \= doorstop);
        (Shape = wedge, O \= flat)
    )), ValidOrientations),
    random_member(Orientation, ValidOrientations).

random_interaction(MaxId, Id, Interaction) :-
    findall(I, interaction(I), Interactions),
    random_member(Template, Interactions),
    assign_random_id(Template, MaxId, Id, Interaction).

assign_random_id(grounded, _, _, grounded).
assign_random_id(touching(_), MaxId, Id, touching(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(pointing(_), MaxId, Id, pointing(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(on_top_of(_), MaxId, Id, on_top_of(T)) :-
    random_between(0, MaxId, T),
    T \= Id.
assign_random_id(inside(_), MaxId, Id, inside(T)) :-
    random_between(0, MaxId, T),
    T \= Id.

item_has_attribute(Attr, item(_,C,S,O,I)) :-
    (
        Attr = C
    ;
        Attr = S
    ;
        % Match vertical group
        (Attr = vertical, (O = upright ; O = upside_down))
    ;
        % Match horizontal group
        (Attr = flat, (O = flat ; O = cheesecake; O = doorstop))
    ;
        % Direct match of orientation
        Attr = O
    ;
        % Interaction check
        nonvar(I),
        I =.. [Attr | _]
    ).

item_has_two_attributes(A1, A2, Item) :-
    A2 \= grounded,
    item_has_attribute(A1, Item),
    item_has_attribute(A2, Item).

count_attribute(Attr, Structure, Count) :-
    include(item_has_attribute(Attr), Structure, Filtered),
    length(Filtered, Count).

count_multiple_attributes(A1, A2, Structure, Count) :-
    include(item_has_two_attributes(A1, A2), Structure, Filtered),
    length(Filtered, Count).

% Special case: grounded or ungrounded
count_interaction_matching_items(QAttr, grounded, Structure, Count) :-
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, Interaction), Structure),
            functor(Interaction, F, _),
            F \= on_top_of,
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, Interaction))
        ),
        Matches),
    sort(Matches, Unique),
    length(Unique, Count).

count_interaction_matching_items(QAttr, ungrounded, Structure, Count) :-
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, on_top_of(_)), Structure),
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, on_top_of(_)))
        ),
        Matches),
    sort(Matches, Unique),
    length(Unique, Count).

% Counts how many unique items have QAttr and are in an interaction with a target with IAttr.
count_interaction_matching_items(QAttr, IAttr, touching, Structure, Count) :-
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, touching(TargetId)), Structure),
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, touching(TargetId))),
            member(item(TargetId, TC, TS, TO, _), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, _))
        ),
        Matches1),
    findall(SourceId,
        (
            member(item(TargetId, TC, TS, TO, touching(SourceId)), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, touching(SourceId))),
            member(item(SourceId, SC, SS, SO, _), Structure),
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, _))
        ),
        Matches2),
    append(Matches1, Matches2, All),
    sort(All, Unique),
    length(Unique, Count).

count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, Count) :-
    InteractionName \= touching,
    findall(SourceId,
        (
            member(item(SourceId, SC, SS, SO, Interaction), Structure),
            Interaction =.. [InteractionName, TargetId],
            item_has_attribute(QAttr, item(SourceId, SC, SS, SO, Interaction)),
            member(item(TargetId, TC, TS, TO, _), Structure),
            item_has_attribute(IAttr, item(TargetId, TC, TS, TO, _))
        ),
        Matches),
    sort(Matches, Unique),
    length(Unique, Count).

has_interaction_attribute(QAttr, grounded, _Structure, item(Id,C,S,O,I)) :-
    item_has_attribute(QAttr, item(Id,C,S,O,I)),
    \+ (I = on_top_of(_)).

has_interaction_attribute(QAttr, ungrounded, Structure, item(_,C,S,O,on_top_of(_))) :-
    member(item(_,C,S,O,on_top_of(_)), Structure),
    item_has_attribute(QAttr, item(_,C,S,O,on_top_of(_))).
% Check if an item has QAttr and an interaction of type InteractionName that leads to another item with IAttr
has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, item(_,C,S,O,I)) :-
    item_has_attribute(QAttr, item(_,C,S,O,I)),
    decode_interaction(I, InteractionName, Target),
    Target \= none, % Not really needed, because if grounded, we dont use this case
    member(item(Target,TC,TS,TO,TI), Structure),
    item_has_attribute(IAttr, item(Target,TC,TS,TO,TI)).

% Decode the interaction to see if it matches the given name and extract the target if any.
decode_interaction(grounded, grounded, none).
decode_interaction(grounded, Name, none) :- Name \= grounded.
decode_interaction(I, Name, T) :-
    I =.. [Name,T], !.
decode_interaction(_, _, none).

% Check different constraints for the interaction between items
interaction_constraint_check(Structure) :-
    %% On_top_of %%
    % Ensure all on_top_of attributes have the right orientation and check if the stacking is valid
    forall(
        member(item(SourceId,_,SourceShape,SourceOrientation,on_top_of(TargetId)), Structure),
        (
            member(item(TargetId, _, TargetShape, TargetOrientation, TargetInteraction), Structure),
            (
                % Grounded cases
                (TargetInteraction = grounded,
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % Pointing cases
                (TargetInteraction = pointing(SourceId),
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % Touching cases
                (TargetInteraction = touching(_),
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = vertical);
                  (TargetShape = pyramid, SourceShape = pyramid, TargetOrientation = SourceOrientation, TargetOrientation = upright)));
                % On top of another item, source is on top of target
                (TargetInteraction = on_top_of(AnotherId),
                 AnotherId \= SourceId,
                 ((TargetShape = block);
                  (TargetShape = wedge, TargetOrientation = cheesecake);
                  (TargetShape = pyramid, TargetOrientation = vertical, SourceOrientation = upright);
                  (TargetShape = pyramid, TargetOrientation = upright, TargetOrientation = SourceOrientation);
                  % Upside-down pyramid stacking inside another upside-down pyramid
                  (TargetShape = pyramid, SourceShape = pyramid,
                   TargetOrientation = upside_down, SourceOrientation = upside_down);

                  % Upside-down pyramid on block
                  (TargetShape = block, SourceShape = pyramid, SourceOrientation = upside_down, TargetOrientation = SourceOrientation)))
            )
        )
    ),
    % Allow a flat block to have up to two items on_top_of it; other items can only have one
    % Currently the rule is turned off: If you want to activate, change the Count of the rule
    forall(
        member(item(TargetId, _, TargetShape, TargetOrientation, _), Structure),
        (
            (TargetShape = block, TargetOrientation = flat ->
                (
                    findall(SourceId, member(item(SourceId, _, _, _, on_top_of(TargetId)), Structure), Sources),
                    length(Sources, Count),
                    % Change to 2, if you want to activate the two objects on top of a flat block rule
                    Count =< 2
                );
                (
                    findall(SourceId, member(item(SourceId, _, _, _, on_top_of(TargetId)), Structure), Sources),
                    length(Sources, Count),
                    Count =< 1
                )
            )
        )
    ),
    % Ensure no loops in the on_top_of chain
    forall(
        member(item(SourceId, _, _, _, on_top_of(_)), Structure),
        \+ has_loop(SourceId, Structure)
    ),
    %% Touching %%
    % Every item can only have max 4 touching items (above = on_top_of, below = none)
    \+ (
        member(item(TargetId,_,_,_,_), Structure),
        findall(SourceId,
                (member(item(SourceId,_,_,_,Interaction), Structure),
                Interaction = touching(TargetId)),
                Sources),
        length(Sources, Count),
        Count > 4
    ),
    % Upside-down pyramids must either be touched by at least 2 others or be on top of an upside-down block
    forall(
        member(item(PyramidId, _, pyramid, upside_down, Interaction), Structure),
        (
            % Option 1: touched by at least 2 valid pieces
            findall(OtherId,
                (
                    member(item(OtherId, _, OtherShape, OtherOrientation, touching(PyramidId)), Structure),
                    (
                        OtherShape = block
                        ;
                        (OtherShape \= block, OtherOrientation = upright)
                    )
                ),
                ValidTouching),
            length(ValidTouching, Count),
            Count >= 2
            ;
            % Option 2: on_top_of an upside-down pyramid stack, base is upside-down block
            Interaction = on_top_of(ParentId),
            upside_down_stack_has_block_base(ParentId, Structure)
        )
    ),
    % Upside-down wedges must be touched by at least 2 other pieces
    forall(
        member(item(WedgeId, _, wedge, upside_down, _), Structure),
        (
            findall(OtherId,
                (
                    member(item(OtherId, _, OtherShape, OtherOrientation, touching(WedgeId)), Structure),
                    (
                        OtherShape = block
                        ;
                        (OtherShape = pyramid, OtherOrientation = upright)
                    )
                ),
                ValidTouching
            ),
            length(ValidTouching, Count),
            Count >= 2
        )
    ),
    % Ensure at least one touching element is grounded via grounded or pointing (all pointing are grounded)
    % Or the touching item is part of a chain of touching items
    forall(
        member(item(SourceId,_,_,_,touching(TargetId)), Structure),
        (member(item(TargetId,_,_,_,grounded), Structure);
        member(item(TargetId,_,_,_,touching(_)), Structure);
        member(item(TargetId,_,_,_,pointing(_)), Structure))
    ),
    %% Pointing %%
    % Only flat, cheesecake, and doorstop items can point (all grounded horizontally)
    % Upright and upside_down items cannot point at all
    forall(
        member(item(SourceId, _, _, _, pointing(TargetId)), Structure),
        (
            (member(item(SourceId, _, _, flat, _), Structure),
             member(item(TargetId, _, _, _, grounded), Structure))
        ;
            (member(item(SourceId, _, _, cheesecake, _), Structure),
             member(item(TargetId, _, _, _, grounded), Structure))
        ;
            (member(item(SourceId, _, _, doorstop, _), Structure),
             member(item(TargetId, _, _, _, grounded), Structure))
        )
    ),
    no_flat_toucher_has_on_top(Structure).

% Ensure no flat piece touches an upside-down pyramid or wedge and has something on top
no_flat_toucher_has_on_top(Structure) :-
    \+ (
        member(item(FlatId, _, _, flat, _), Structure),

        % Check if flat piece is touching an upside-down piece
        (
            member(item(_, _, Shape, upside_down, touching(FlatId)), Structure)
            ;
            member(item(_, _, Shape, upside_down, _), Structure),
            member(item(FlatId, _, _, _, touching(UpsideDownId)), Structure),
            member(item(UpsideDownId, _, Shape, upside_down, _), Structure)
        ),
        (Shape = pyramid; Shape = wedge),

        % And has something on top
        member(item(_, _, _, _, on_top_of(FlatId)), Structure)
    ).

% Helper functions for checking, if on_top_of creates a loop!
has_loop(StartId, Structure) :-
    has_loop_helper(StartId, StartId, Structure, []).

% If the current ID has already been visited, theres a loop
has_loop_helper(_, CurrentId, _, Visited) :-
    member(CurrentId, Visited),
    !.
has_loop_helper(StartId, CurrentId, Structure, Visited) :-
    member(item(CurrentId, _, _, _, on_top_of(NextId)), Structure),
    has_loop_helper(StartId, NextId, Structure, [CurrentId|Visited]).

% Recursively walk the on_top_of chain to find the base of an upside-down pyramid stack.
upside_down_stack_has_block_base(ItemId, Structure) :-
    member(item(ItemId, _, block, upside_down, _), Structure).

upside_down_stack_has_block_base(ItemId, Structure) :-
    member(item(ItemId, _, pyramid, upside_down, on_top_of(ParentId)), Structure),
    upside_down_stack_has_block_base(ParentId, Structure).

attribute_type(A, color) :- color(A), !.
attribute_type(A, shape) :- shape(A), !.
attribute_type(A, orientation) :- orientation(A), !.
%%% Rules %%%
% Check predicates (pure checks, no generation)
zero(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    C =:= 0.

zero(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C =:= 0.

at_least(Attr, N, Structure) :-
    count_attribute(Attr, Structure, C),
    C >= N.

at_least(A1, A2, N, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C >= N.

% Generic version for any interaction that involves a target item
at_least_interaction(QAttr, IAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count >= N.

at_least_interaction(QAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count >= N.

all(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    length(Structure, L),
    C =:= L.

all(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    length(Structure, L),
    C =:= L.

exactly(Attr, N, Structure) :-
    count_attribute(Attr, Structure, C),
    C =:= N.

exactly(A1, A2, N, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C =:= N.

% Generic version for any interaction that involves a target item
exactly_interaction(QAttr, IAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, IAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count =:= N.

exactly_interaction(QAttr, InteractionName, N, Structure) :-
    findall(Item,
        (member(Item, Structure),
        has_interaction_attribute(QAttr, InteractionName, Structure, Item)),
        Filtered),
    length(Filtered, Count),
    Count =:= N.

exclusively(Attr, Structure) :-
    % For every item in the structure, Attr must hold
    forall(member(Item, Structure), item_has_attribute(Attr, Item)),

    % Ensure no other attribute of the same type appears
    (
        color(Attr) ->
            findall(C, color(C), Colors),
            exclude(==(Attr), Colors, OtherColors),
            \+ (member(Other, OtherColors), member(Item, Structure), item_has_attribute(Other, Item))
    ;
        shape(Attr) ->
            findall(S, shape(S), Shapes),
            exclude(==(Attr), Shapes, OtherShapes),
            \+ (member(Other, OtherShapes), member(Item, Structure), item_has_attribute(Other, Item))
    ;
        orientation(Attr) ->
            findall(O, orientation(O), Orients),
            exclude(==(Attr), Orients, OtherOrients),
            \+ (member(Other, OtherOrients), member(Item, Structure), item_has_attribute(Other, Item))
    ;
        true
    ).

more_than(A1, A2, Structure) :-
    count_attribute(A1, Structure, C1),
    count_attribute(A2, Structure, C2),
    C1 > C2.

odd_number_of(Structure) :-
    length(Structure, L),
    1 is L mod 2.

even_number_of(Structure) :-
    length(Structure, L),
    L \= 0,
    0 is L mod 2.

odd_number_of(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    1 is C mod 2.

odd_number_of(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    1 is C mod 2.

odd_number_of_interaction(QAttr, IAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, C),
    1 is C mod 2.

odd_number_of_interaction(QAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, InteractionName, Structure, C),
    1 is C mod 2.

even_number_of(Attr, Structure) :-
    count_attribute(Attr, Structure, C),
    C \= 0,
    0 is C mod 2.

even_number_of(A1, A2, Structure) :-
    count_multiple_attributes(A1, A2, Structure, C),
    C \= 0,
    0 is C mod 2.

even_number_of_interaction(QAttr, IAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, IAttr, InteractionName, Structure, C),
    C \= 0,
    0 is C mod 2.

even_number_of_interaction(QAttr, InteractionName, Structure) :-
    count_interaction_matching_items(QAttr, InteractionName, Structure, C),
    C \= 0,
    0 is C mod 2.

either_or(N1, N2, Structure) :-
    length(Structure, L),
    (L =:= N1; L =:= N2).

length_(N1, Structure) :-
    length(Structure, L),
    (L =:= N1).

same_amount(A1, A2, Structure) :-
    count_attribute(A1, Structure, C1),
    count_attribute(A2, Structure, C2),
    C1 \= 0,
    C2 \= 0,
    C1 =:= C2.

and([]).
and([length(_)|Cs]) :- !, and(Cs).  % length(N) is a size constraint, not a callable check
and([C|Cs]) :-
    call(C),
    and(Cs).

or([C|_]) :-
    call(C).
or([_|Cs]) :-
    or(Cs).

all_three_shapes(Structure) :-
    findall(Shape, shape(Shape), Shapes),
    forall(
        member(S, Shapes),
        (
            count_attribute(S, Structure, C),
            C > 0
        )
    ).

all_three_colors(Structure) :-
    findall(Color, color(Color), Colors),
    forall(
        member(C, Colors),
        (
            count_attribute(C, Structure, Count),
            Count > 0
        )
    ).

odd(N) :- 1 is N mod 2.
even(N) :- 0 is N mod 2.
ge(Min, N) :- N >= Min.
eq(X, Y) :- X =:= Y.

exclude_list(List, Excluded, Result) :-
    exclude(in_list(Excluded), List, Result).

in_list(L, X) :- member(X, L).