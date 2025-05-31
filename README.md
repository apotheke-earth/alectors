# alectors

`alectors` is a library providing transformer-based rl agents for nlp tasks, that is extremely customizable, but also comes with sane defaults.

> source code: [https://erga.apotheke.earth/apotheke/alectors](https://erga.apotheke.earth/apotheke/alectors)  
> license: [GPLv3 or later](https://gnu.org/licenses/gpl-3.0.html)  

## Why "alectors"?

The word lector has deep etymological roots in language. Derived from the Latin *legere* ("to read"), it originally referred to someone who read aloud, to an audience, students, or even in religious ceremonies.  
The term nowdays survives mostly in an academic setting, where a lecturer is typically tied to teaching, or discussing ideas with an audience. Typical modern uses of the word are "lecturer" in English, "λέκτορας" in Greek, and "lektor" in Polish.  
However, adding the prefix α- changes the word's meaning entirely. An *alector* (*ἀλέκτωρ*) means "cock" (rooster) in Greek. This juxtaposition s very funny, hence the name.

## Description

Modern NLP solutions like GPTs and BERTs have made great strides in language processing and generation, however they fall short of actual decision making.

As an example, consider an LLM trying to play a game of chess.

While it may be able to make valid moves, and even provide a justification, it still lacks any *true* capacity to calculate the optimal position; it is a word generator, and so the best thing it can do is convince itself that it makes sense.
It lacks a reward incentive during training, and even worse, rather LLMs rely on static token distributions without real-time feedback.

`alectors` tries to address this gap by shifting the focus to reinforcement learning in place of supervised pretraining. An *alector* learns to generate *actions* based on natural language.

The way we approach this is by using pretrained encoders in specialized language environments.

### Mechanism

In essence, we can describe any environment using natural language; some environments lend themselves to this type of description better, and this description of the state is then given to a pretrained encoder, that parses it and spits out a layer of vector embeddings[^1].

This *vector representation of the natural language description of the state* is then passed to the agent, and used in traditional rl ways (ppo, sac, etc)

The agent therefore trains on natural language reinforcement learning, without the need of any pretraining/finetuning as is the case for LLMs.

## Supported Architectures

The currently supported architecture is PPO. Plans exist to include SAC, GRPO, and maybe DDQN. As we add them to `telestes` they will be almost automatically imported here[^2].

[^1]: for our purpose we care about the token embeddings, so we throw the [CLS] token away.
[^2]: modulo some tweaks in `learn()` and `choose_action()`, to include the encoder in the functions.
