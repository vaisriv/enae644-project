#import "@preview/bamdone-ieeeconf:0.1.3": ieee

#show: ieee.with(
    title: [ENAE644 Term Project],
    // TODO: add abstract
    abstract: [
        // FIXME:
        #lorem(100)
    ],
    authors: (
        (
            given: "Vai",
            surname: "Srivastava",
            email: [vaisriv],
            affiliation: 1,
        ),
    ),
    affiliations: (
        (
            name: [the Department of Aerospace Engineering, University of Maryland],
            address: [College Park, MD 20742, The United States of America],
            email-suffix: [umd.edu],
        ),
    ),
    index-terms: (),
    bibliography: bibliography(
        "references.yaml",
        style: "ieee",
        title: text(10pt)[References],
    ),
    draft: false,
    disclaimer: [],
    paper-size: "us-letter",
)

// TODO:
// - [ ] add abstract
// - [ ] formal problem definition of the problem you are solving
// - [ ] make sure to pick the exact methods (specific algorithms and implementation processes for each agent) that you plan to use
// - [ ] please document current progress on your implementation (what you have done so far/are currently doing)
// - [ ] list the experiments that you plan to run/currently running experiments

= Introduction <introduction>

The goal of this project is to implement and evaluate algorithms from two complementary sub-fields of adversarial motion planning. The first is _deceptive motion planning_, which is concerned with generating trajectories that conceal the agent's true objective from an observer @nichols2022adversarial @xu2020single. The second is _goal identification and interception_, which seeks to infer an observed agent's hidden objective and plan a trajectory to intercept it @zeng2023recognition @tastan2012learning. Rather than studying these in isolation, the project will pit a deceptive planner against an interception-focused planner in a shared environment, providing a direct adversarial evaluation of each algorithm's strengths and limitations. Both agents will employ planning methods at least as sophisticated as RRT\*, incorporating learning-based and game-theoretic components on top of their respective base planners.

= Motivation <motivation>

Deceptive motion planning has direct applications in video game AI, where non-player characters must exhibit unpredictable yet purposeful behavior @tastan2012learning @zeng2023recognition. However, the relevance of this problem extends well beyond entertainment into real-world domains such as strategic logistics, privacy-preserving navigation, and military operations. The ability to move deceptively can be the difference between mission success and failure @xu2020single @nichols2022adversarial, while the complementary problem of identifying and intercepting deceptive agents is equally critical: security and defense systems must be capable of recognizing when an observed agent is behaving deceptively and responding accordingly @zeng2023recognition @xu2019goal. Studying the interplay between deception and counter-deception will provide insight into the robustness of each approach and reveal failure modes that would not surface when evaluating either in isolation.

It is important to note that prior work has explored this coupling in the context of autonomous vehicles @espinoza2022deep and multi-agent navigation @netter2024motion @lecleach2021lucidgames, but an adversarial pitting in a strategy-game context remains underexplored.

= Related Work <related-work>

== Deceptive Motion Planning <deceptive-motion-planning>

A central challenge in deceptive motion planning is balancing path cost against unpredictability. Nichols et al.~@nichols2022adversarial address this trade-off with Adversarial RRT\*, a sampling-based planner that incorporates a learned deception cost into the RRT\* objective. Their approach uses a recurrent neural network as a surrogate observer and penalizes trajectories that the network can classify correctly, achieving low observer accuracy while keeping path length near-optimal. Xu et al.~@xu2020single take a complementary approach grounded in probabilistic goal recognition, formalizing deceptive path planning as a mixed-integer program that maximizes goal ambiguity with respect to an observer's belief distribution. Both methods assume a passive observer and evaluate deception in isolation rather than against an active adversary, which is the gap this project aims to address.

== Goal Recognition and Interception <goal-recognition-and-interception>

On the opposing side of the problem, several works focus on inferring an agent's hidden objective and acting on that inference. Xu and Yin~@xu2019goal introduce _relative goal uncertainty_, an entropy-based metric that quantifies the goal-related information contained in each action, and show how it can be used to control the goal identification process from either side of the interaction. Zeng and Xu~@zeng2023recognition extend goal recognition to explicitly account for deceptive agents by using inverse reinforcement learning to recover a behavioral model from the deceiver's historical data, then constructing a game-theoretic interference strategy. In the domain of first-person shooter games, Tastan et al.~@tastan2012learning learn player-specific motion models via inverse reinforcement learning and use particle-filter-based prediction to intercept opponents on partially occluded maps. These works provide the algorithmic foundations for the interceptor agent in this project, particularly the combination of learned behavioral models with online prediction.

== Game-Theoretic and Interactive Planning <game-theoretic-and-interactive-planning>

A related body of work frames multi-agent planning as a dynamic game in which each agent's strategy depends on the others'. Le Cleac'h et al.~@lecleach2021lucidgames propose LUCIDGames, which pairs an unscented Kalman filter for online estimation of other agents' cost functions with a receding-horizon game-theoretic planner, demonstrating real-time performance in autonomous driving scenarios. Espinoza et al.~@espinoza2022deep tightly couple prediction and planning through a game-theoretic MPC that uses a multi-agent neural network policy as its predictive model, producing interactive behaviors between an autonomous vehicle and surrounding traffic. While both of these works operate in the autonomous driving domain, they demonstrate the value of jointly reasoning about one's own plan and other agents' likely responses, directly informing the adversarial evaluation framework in this project.

== Adversary-Aware Motion Planning <adversary-aware-motion-planning>

Most closely related to the present work, Netter and Vamvoudakis~@netter2024motion propose a motion planning framework in which a player agent navigates a multi-agent environment while simultaneously identifying and avoiding potential adversaries using Gaussian process classification. Their method includes real-time replanning to avoid likely adversarial agents and distinguishes adversaries from benign agents to prevent unnecessary evasive maneuvers. This project builds on a similar premise but inverts the emphasis: rather than treating interception avoidance as a byproduct of adversary classification, the deceptive agent here will actively minimize observer accuracy as a first-class planning objective, while the interceptor will be a fully autonomous adversary rather than a fixed behavioral model.

// TODO: formal problem definition of the problem you are solving
= Formal Problem Definition <formal-problem-definition>

// FIXME:
#lorem(200)

= Methodology <methodology>

// TODO: make sure to pick the exact methods (specific algorithms and implementation processes for each agent) that you plan to use
== Overview <overview>

The project is implemented in Python, chosen for its extensive ecosystem of machine learning and scientific computing libraries. In particular, we use JAX (via the Equinox library) as the primary framework for differentiable programming, automatic differentiation, and GPU-accelerated computation.

The core of the project involves two competing agents in a continuous two-dimensional workspace. The first agent employs a deceptive motion planning algorithm building on Adversarial RRT\* @nichols2022adversarial, which augments the sampling-based RRT\* planner with a learned deception cost to generate trajectories that minimize an observer's ability to infer the agent's true goal. Entropy-based deceptive planning techniques @xu2020single @xu2019goal inform the design of the deception objective. The second agent utilizes an identification and interception strategy, drawing on techniques from inverse reinforcement learning and game-theoretic prediction @zeng2023recognition @tastan2012learning @lecleach2021lucidgames, with the objective of recognizing the deceptive agent's true goal and planning an intercept trajectory in real time. The approach of coupling adversary identification with reactive planning follows the framework proposed by Netter and Vamvoudakis~@netter2024motion, adapted to the adversarial evaluation setting. Both agents thus operate well beyond basic shortest-path planning: the deceptive agent layers a learned deception metric on top of RRT\*, while the interceptor combines online goal inference with reactive replanning.

Of note, despite several of the referenced algorithms being originally developed for three-dimensional environments (e.g., first-person shooter game maps @tastan2012learning), we work in a continuous two-dimensional workspace for ease of development and in the interest of completing the project within the semester. The simulation environment is custom-built in Python, providing full control over the experimental setup and enabling systematic evaluation of each algorithm's performance under varying conditions.

// TODO: please document current progress on your implementation (what you have done so far/are currently doing)
== Implementation <implementation>

// FIXME:
#lorem(200)

// TODO: list the experiments that you plan to run/currently running experiments
== Procedure <procedure>

// FIXME:
#lorem(200)
