import torch.optim as optim

from tqdm import tqdm

from alectors.ppo import Agent

hparams = dict(
    actor_hparams = dict(
        optimizer=optim.Adam,
        optimizer_hparams = dict(
            betas=(0.9, 0.9)
        )
    ),
    critic_hparams = dict(
        optimizer=optim.Adam
    )
)
agent = Agent(
    output_dims=4,
    batch_size=10,
    epochs=2,
    verbose=False,
    load=False,
    **hparams
)

agent.save()

full_state = "You are inside a maze, to escape you must go: Up, Up, Left, Down.".split()
state = ""
for i in tqdm(range(len(full_state))):
    # Test varying input sizes
    state += full_state[i]
    action, probs, vals = agent.choose_action(state)
    agent.store(state, action, probs, vals, 0.11, False, False)

hparams = dict(
    actor_hparams = dict(
        optimizer=optim.Adam,
        optimizer_hparams = dict(
            betas=(0.9, 0.9)
        )
    ),
    critic_hparams = dict(
        optimizer_name='adam'
    )
)
agent = Agent(
    output_dims=4,
    batch_size=4,
    epochs=2,
    verbose=True,
    load=True,
    **hparams
)

full_state = "You are inside a maze, to escape you must go: Up, Up, Left, Down.".split()
state = ""
for i in tqdm(range(len(full_state))):
    # Test varying input sizes
    state += full_state[i]
    action, probs, vals = agent.choose_action(state)
    agent.store(state, action, probs, vals, 0.11, False, False)


loss = agent.learn()

