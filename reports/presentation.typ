#import "@preview/touying:0.7.3": *
#import themes.university: *
#import "@preview/numbly:0.1.0": numbly

#show: university-theme.with(
    aspect-ratio: "16-9",
    align: horizon,
    config-info(
        title: [Adversarial Motion Planning],
        subtitle: [ENAE644 Final Project],
        author: [Vai Srivastava],
        date: datetime.today(),
        institution: [University of Maryland],
    ),
    config-colors(
        primary: rgb("#E03B45"),
        secondary: rgb("#9F9F9F"),
        tertiary: rgb("#FACC00"),
        neutral-lightest: rgb("#ffffff"),
        neutral-darkest: rgb("#000000"),
    ),
)


#set heading(numbering: numbly("{1}.", default: "1.1"))
#set text(size: 18pt)
#show math.equation: set text(size: 16pt)

#title-slide(logo: image("../assets/informal-seal.png", width: 7.5%))

= Outline <touying:hidden>

#components.adaptive-columns(outline(title: none, indent: 1em))


= Problem Statement

== A Two-Agent Adversarial Scenario

#cols(columns: (1.2fr, 1fr), gutter: 1em)[
    - Two competing planners share a continuous 2D workspace.

    - *Agent D* (deceptive) --- must reach a *hidden* goal while concealing its intent from any observer.

    - *Agent I* (interceptor) --- must *infer* Agent D's goal and *intercept* before D arrives.

    - Both planners go beyond plain RRT\*: they layer learning- and game-theoretic components on top of sampling-based search.

    - Goal: a *direct adversarial pitting* --- one deceiver vs. one learning interceptor in the same sandbox.
][
    #image("../outputs/figures/trajectories.png", height: 8cm)
]

#speaker-note[
    - Most prior work studies *one* side at a time; this project pits them directly against each other.
    - The image is a sneak peek at the experimental scenario --- we'll dissect it in the results.
    - Land the framing line: "two planners, *competing*, in the same workspace."

    *Acronyms on this slide:*
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically-optimal variant) --- a sampling-based path planner.
]


= Motivation & Related Work

== Why Adversarial Motion Planning?

*Applications.* Game AI, privacy-preserving navigation, autonomous-vehicle interaction, defense and security.

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Deceptive planning*
    - Nichols et al., 2022 --- *Adversarial RRT\** with a learned observer.
    - Xu et al., 2020 --- MILP for goal-ambiguity maximization.
][
    *Goal recognition + interception*
    - Tastan et al., 2012 --- IRL + particle filter (FPS games).
    - Zeng & Xu, 2023 --- IRL + game-theoretic counter-deception.
]

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Game-theoretic / interactive planning*
    - LucidGames (Le Cleac'h et al., 2021).
    - Espinoza et al., 2022 --- game-theoretic MPC + neural prediction.
][
    *Adversary-aware*
    - Netter & Vamvoudakis, 2024 --- GP-classifier-based adversary avoidance.
]

*Gap addressed.* Direct head-to-head evaluation of a *deceptive* planner against a *learning* interceptor in a strategy-game-style setting.

#speaker-note[
    - ~10s per cluster --- the audience came for the formulation, not the lit review.
    - Punchline: nobody's run these two algorithms head-to-head in one sandbox.

    *Acronyms on this slide:*
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically optimal).
    - *MILP* = Mixed-Integer Linear Program (an optimization formulation).
    - *IRL* = Inverse Reinforcement Learning (recover a reward from demos).
    - *FPS* = First-Person Shooter (a video-game genre).
    - *MPC* = Model Predictive Control (replanning over a finite horizon).
    - *GP* = Gaussian Process (a non-parametric Bayesian classifier here).
]


= Technical Formulation

== Workspace and Agents

*Workspace.* Bounded 2D region $cal(W) subset RR^2$, static obstacles $cal(O) subset cal(W)$, free space $cal(W)_"free" = cal(W) without cal(O)$.

*Agents.* Positions $x_D (t), x_I (t) in cal(W)_"free"$; trajectories $xi_D, xi_I : [0, T] -> cal(W)_"free"$ subject to kinodynamic constraints $xi in Xi$.

*Goals.*
- $cal(G)_D = {g_1, dots, g_M}$ --- *candidate* goals, known to both agents.
- $g^* in cal(G)_D$ --- the *true* goal, known *only* to Agent D.

*Belief.* (Agent I's distribution over $cal(G)_D$) $b_t : cal(G)_D -> [0,1]$, with $sum_(g) b_t (g) = 1$.

#speaker-note[
    - The class hasn't seen this notation --- walk through every symbol.
    - "Belief" is just a probability distribution over which goal D might be heading to.
    - Stress: D and I share the *candidate* set, but only D knows which one is real.

    *Acronyms on this slide:* none --- only mathematical symbols.
]


== Agent D's Objective: Path vs. Deception

Agent D solves a *weighted multi-objective* problem:

$
    xi_D^* = arg min_(xi_D in Xi_D) [thin alpha thin J_"path" (xi_D) + (1 - alpha) thin J_"deception" (xi_D) thin]
$

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *$J_"path"$ --- path cost*
    $ J_"path" (xi_D) = integral_0^T ||dot(xi)_D (t)|| dif t $
    Total length / control effort.
][
    *$J_"deception"$ --- "give-away"*
    $ J_"deception" (xi_D) = f_theta (xi_D)(g^*) $
    Probability a *learned classifier* $f_theta$ assigns to the *true* goal --- we want this *low*.
]

*Trade-off knob* $alpha in [0,1]$: 1 → optimal path, 0 → max deception.

*Solver.* Adversarial RRT\* (uses the combined cost above as its rewire metric).

#speaker-note[
    - "Surrogate observer": $f_theta$ is a stand-in for any external watcher trying to guess D's goal from a partial path. Penalizing high $f_theta(xi)(g^*)$ means "make the watcher uncertain."
    - Note: at $alpha = 1$ this *recovers plain RRT\**.

    *Acronyms on this slide:*
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically optimal).
]


== Agent I: Goal Inference + Game-Theoretic Planning

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Inference (offline → online)*

    *Offline --- Inverse RL.* Given expert demonstrations of D, recover a reward $R_D (s, a)$ that explains them. Plain English: "given good behavior, infer the cost that produced it."

    *Online --- Particle Filter.* Maintain $N$ weighted goal-hypotheses; re-weight each by Bayes' rule using $R_D$ as the likelihood:
    $ b_(t+1)(g) prop P(xi_D [t,t+1] | g) thin b_t (g) $

    MAP estimate: $hat(g)(t) = arg max_g b_t (g)$.
][
    *Planning --- Game-Theoretic MPC*

    At every timestep, solve a *finite-horizon* optimization:
    $ min_(xi_I [t, t+H]) J_I (xi_I, hat(xi)_D) $

    $hat(xi)_D$ = belief-weighted predicted trajectory of D.
    $
        J_I = ||xi_I (T_"int") - hat(g)||^2 + integral_t^(t+H) ||dot(xi)_I||^2 dif tau
    $

    Execute one step, *replan*, repeat.
]

#speaker-note[
    - IRL: emphasize this is *offline* --- we use prior demonstrations of D before the game starts.
    - Particle filter: $N$ guesses, weighted by how plausible each is given what's been observed so far.
    - MPC: short-horizon optimal control, replan every tick --- no commitment to a stale prediction.

    *Acronyms on this slide:*
    - *IRL* / *RL* = Inverse Reinforcement Learning / Reinforcement Learning --- IRL recovers a reward function $R_D$ from expert demonstrations.
    - *MAP* = Maximum A Posteriori (the most-probable goal given the belief).
    - *MPC* = Model Predictive Control (finite-horizon optimization, replanned each step).
]


== Adversarial Game Structure

A *two-player dynamic game with asymmetric information*:

- Agent D knows $g^*$; Agent I does *not*.
- *Stackelberg-style.* D leads, I responds to its observations.

*Termination conditions:*

- *Agent D wins:* $||x_D (t) - g^*|| < 0.5$ (reaches its true goal).
- *Agent I wins:* $||x_I (t) - x_D (t)|| < 0.5$ (catches D).
- *Timeout* at 60 s.

*Evaluated empirically* via simulation (no closed-form equilibrium).

#speaker-note[
    - Two key ideas: *asymmetric info* and the explicit *win conditions*.
    - We chose simulation over equilibrium analysis because the IRL/particle-filter dynamics are not analytically tractable.

    *Acronyms (mentioned verbally):*
    - *IRL* = Inverse Reinforcement Learning.
]


= Implementation

== System Architecture

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Stack.* Python + JAX + Equinox --- automatic differentiation, GPU acceleration, pytree-based trajectory representation.

    *Agent D pipeline.*
    - GRU recurrent network → surrogate observer
    - Deception cost $f_theta (xi)(g^*)$
    - Adversarial RRT\* (rewires by combined cost)

    *Agent I pipeline.*
    - 3-layer MLP → reward $R_D$ (contrastive IRL)
    - Particle filter (200 particles) → belief tracker
    - Adam-based gradient-descent MPC
][
    *Two-stage pipeline.*
    - *Offline:* train both networks on synthetic demonstrations, serialize weights to disk.
    - *Online:* load checkpoints, run adversarial sim, log trajectories + metrics + figures.

    *Performance details.*
    - Observer input truncated to last 20 positions to avoid JIT recompiles on growing arrays.
    - RRT\* rewiring uses path-cost only (skip the per-neighbor observer call).
]

#speaker-note[
    - This slide is a *map*, not a deep dive --- one sentence per box.
    - JAX/Equinox = a differentiable functional ML framework, in case anyone asks.

    *Acronyms on this slide:*
    - *JAX* = a NumPy-compatible numerical-computing library with autodiff + GPU; *Equinox* = JAX-based neural-network library.
    - *GPU* = Graphics Processing Unit (used here for accelerated tensor math).
    - *GRU* = Gated Recurrent Unit (a type of *RNN* = Recurrent Neural Network).
    - *MLP* = Multi-Layer Perceptron (a feedforward neural network).
    - *IRL* = Inverse Reinforcement Learning.
    - *MPC* = Model Predictive Control.
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically optimal).
    - *JIT* = Just-In-Time compilation (JAX traces + compiles functions on first call).
    - *Adam* = a gradient-descent optimizer (Adaptive Moment Estimation).
]


== IRL Expert Demonstrations

#cols(columns: (1fr, 1fr), gutter: 1em)[
    #image("../outputs/demos/irl_demonstrations.png", height: 8cm)
][
    *Training input for Agent I's reward network.*

    - 200 *expert paths* generated by a *goal-directed RRT* --- no deception, just direct routing toward one of the three candidate goals.
    - Color-coded by target goal (10 paths/goal shown).
    - Demonstrations start from random positions and respect the central-obstacle collision constraint.
    - *Contrastive IRL* uses these: for each $(s, a)$ in a demo, push the learned reward $R_D$ above 8 random alternative actions.
]

#speaker-note[
    - Each colored cluster = paths to one of the three corner goals, all routed around the obstacle.
    - The point: this is what "expert behavior" looks like to Agent I --- no deception baked in, just goal-directed routing.
    - Contrast with the *deceptive* trajectory we'll see in the results.

    *Acronyms on this slide:*
    - *IRL* = Inverse Reinforcement Learning (here, the *contrastive* variant).
    - *RRT* = Rapidly-exploring Random Tree (the goal-directed variant --- no asterisk because we don't need optimality for the demos).
]


== Experimental Setup

*Setup.* $10 times 10$ workspace, single circular obstacle (radius 1) at $(5,5)$. Goals at $(9,9), (9,1), (1,9)$. D starts $(1,1)$, I starts $(1,9)$; true goal $g^* = (9,9)$.

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Primary trial.*
    - $alpha = 0.5$
    - 2000 Adv. RRT\* iters, step 0.5
    - Particle filter: 200 particles
    - MPC: horizon 15, 50 grad steps
    - $Delta t = 0.1$ s
][
    *Ablation.* (planning-only)
    - $alpha in {0, 0.25, 0.5, 0.75, 1}$
    - 300 RRT\* iters per $alpha$
    - Same observer reused
    - Measures: obs. accuracy, path ratio
]

#speaker-note[
    - Mental picture: D bottom-left, I top-left, true goal top-right --- they both have a corner-to-corner job.
    - The three goals are in *different quadrants* --- flag this for the discussion later; it's the punchline.

    *Acronyms on this slide:*
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically optimal).
    - *MPC* = Model Predictive Control (horizon = look-ahead in steps).
]


= Results

== Primary Trial --- Trajectories

#cols(columns: (1.2fr, 1fr), gutter: 1em)[
    #image("../outputs/figures/trajectories.png", height: 8cm)
][
    *Outcome.* Agent D wins at $t = 2.30$ s.

    *Path length ratio.*
    $J_"path" (xi_D^*) \/ J_"path" (xi_D^"opt") = 1.124$ --- about 12% above optimal (a modest detour).

    *Interception.* Agent I never closes within the 0.5-unit threshold.

    *Note:* The deceptive trajectory swings northwest before committing to $(9,9)$.
]

#speaker-note[
    - Point out the central-obstacle detour and the slight feint upward.
    - "Modest" detour --- only 12% above optimal. As we'll see, *not* enough to fool the observer.

    *Acronyms on this slide:* none.
]


== Inter-Agent Distance and Goal Belief

#cols(columns: (1fr, 1fr), gutter: 1em)[
    #image("../outputs/figures/distance_over_time.png", height: 6cm)

    *Distance.* Min approach 2.50 units --- *5x* the 0.5-unit interception threshold.
][
    #image("../outputs/figures/belief_evolution.png", height: 6cm)

    *Belief.*
    - *Particle filter*: converges to true goal in $approx 1.0$ s.
    - *RNN observer*: assigns probability *1.000* from frame 1.
]

#speaker-note[
    - The headline tension: Agent D *wins* the chase, but *fails on its own deception terms* --- the RNN was never confused.
    - Particle filter converges *slower* than the RNN: I's belief is correct, but I still can't physically catch up in time.

    *Acronyms on this slide:*
    - *RNN* = Recurrent Neural Network --- here, the GRU-based surrogate observer that classifies which goal Agent D is heading to.
]


== Deception-Weight Ablation

#image("../outputs/figures/alpha_sweep_comparison.png", height: 6cm)

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Observer accuracy:* *1.000 for every* $alpha in {0, 0.25, 0.5, 0.75, 1}$.

    *Path length ratio:* 1.06 – 1.20, *no* monotone trend with $alpha$.
][
    *Implication.* Tuning $alpha$ buys you *zero* additional deception in this workspace --- every dollar spent on detours is *wasted*. The deception layer is effectively *inert*.
]

#speaker-note[
    - This is the most surprising result of the project --- pause for a beat after stating "1.000 for every $alpha$."
    - Set up the discussion: *why* is the observer never fooled?

    *Acronyms on this slide:* none --- only the Greek letter $alpha$ (the deception-weight parameter from the Agent D objective).
]


= Discussion

== Why the Deception Layer Was Inert

*Goal-set geometry dominated outcomes, not the deception parameterization.*

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Goal layout matters:*
    - Three *corner* goals → every trajectory commits to a quadrant within a few steps.
    - The RNN observer saturates at perfect classification *regardless of $alpha$*.
    - Contrast: Nichols et al. arrange goals on a *common arc* --- long shared prefix, deception works ($approx 10%$ observer accuracy).
][
    *Interceptor failure mode:*
    - Belief converges *correctly* in 1 s, but...
    - MPC horizon (15 steps) + short episode (2.3 s) → I can't physically catch up in time.
    - IRL training cut loss 56%, but the per-step likelihoods weren't sharp enough to drive aggressive interception.
]

#speaker-note[
    - This is the *thesis* of the talk --- say it audibly.
    - Land the line: "geometry of the workspace is a first-class design variable, not just stage decoration."

    *Acronyms on this slide:*
    - *RNN* = Recurrent Neural Network (the surrogate observer).
    - *MPC* = Model Predictive Control (horizon = number of look-ahead steps).
    - *IRL* = Inverse Reinforcement Learning.
]


= Conclusion

== Summary and Takeaways

#cols(columns: (1fr, 1fr), gutter: 1em)[
    *Built (JAX/Equinox).*
    - Agent D: GRU observer + Adversarial RRT\*.
    - Agent I: contrastive IRL + particle filter + game-theoretic MPC.

    *Found.* Agent D *won* (2.30 s, $approx 12%$ over optimal); but the deception layer was *inert* --- observer accuracy 1.000 for *every* $alpha$ tested.
][
    *Takeaway.* *Goal-set geometry is a first-class design variable* in adversarial motion-planning evaluation.

    *Future work.* Ambiguous goal layouts, 3D, randomized trials, max-entropy IRL, longer MPC horizon, closed-loop D replanning.
]

#speaker-note[
    - Land hard on the "first-class design variable" line --- that's the contribution of the project.
    - Open the floor to questions; expected ones: "what about more goals?", "why didn't you use 3D?", "could the interceptor be smarter?"

    *Acronyms on this slide:*
    - *JAX* / *Equinox* = the autodiff + neural-net libraries used.
    - *GRU* = Gated Recurrent Unit (a type of RNN).
    - *RRT\** = Rapidly-exploring Random Tree (asymptotically optimal).
    - *IRL* = Inverse Reinforcement Learning.
    - *MPC* = Model Predictive Control.
]

== Refernces

#bibliography(
    "references.yaml",
    style: "ieee",
    title: text(10pt)[References],
)
