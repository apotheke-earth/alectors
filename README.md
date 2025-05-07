
# alectors

*alectors* is a library providing transformer-based rl agents for nlp tasks, that is extremely customizable, but also comes with sane defaults.

>source code: [erga.apotheke.earth/aethrvmn/alectors](https://erga.apotheke.earth/aethrvmn/alectors)  
> license: [Don't Be Evil License 1.1 (DBEL 1.1)](https://apotheke.earth/license)  
> [read the full license here](https://apotheke.earth/license)  

## Why "alectors"?

The word lector has deep etymological roots in language. Derived from the Latin *legere* ("to read"), it originally referred to someone who read aloud, to an audience, students, or even in religious ceremonies.  
The term nowdays survives mostly in an academic setting, where a lecturer is typically tied to teaching, or discussing ideas with an audience. Typical modern uses of the word are "lecturer" in English, "λέκτορας" in Greek, and "lektor" in Polish.  
However, adding the prefix α- changes the word's meaning entirely. An *alector* (*ἀλέκτωρ*) means "cock" (rooster) in Greek. This juxtaposition s very funny, hence the name.

## Reasoning

Modern NLP solutions like GPTs and BERTs have made great strides in language processing and generation, however they fall short of actual decision making.
As an example, consider an LLM trying to play a game of chess.  
While it may be able to make valid moves, and even provide a justification, it still lacks any *true* capacity to calculate the optimal position; it is a word generator, and so the best thing it can do is convince itself that it makes sense.
It lacks a reward incentive during training, and even worse, rather LLMs rely on static token distributions without real-time feedback.

*alectors* tries to address this gap by shifting the focus to active learning through reinforcement. An *alector* doesn’t just learn to generate language; it learns to generate actions based on natural language, using reinforcement learning solutions. 

## Supported Architectures

The currently supported architecture is PPO. Plans exist to include SAC, GRPO, and maybe DDQN.
