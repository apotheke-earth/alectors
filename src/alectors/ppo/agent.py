import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from telestes import PPOAgent

class Agent(PPOAgent):
    def __init__(
        self,
        output_dims: int|None,
        gamma: float = 0.99,
        epochs: int = 10,
        gae_lambda: float =0.95,
        policy_clip: float = 0.1,
        entropy: float = 0.001,
        batch_size: int = 64,
        encoder_name: str = 'bert-base-uncased',
        tokenizer_name: str = 'bert-base-uncased',
        actor_ridge: float = 0,
        actor_coefficient: float = 1.,
        actor_clip_grad_norm: float = None,
        critic_ridge: float = 0,
        critic_coefficient: float = 1.,
        critic_clip_grad_norm: float = None,
        name: str = 'alector',
        verbose: bool = False,
        load: bool = False,
        chkpt_dir: str = 'chkpts',
        device: str = 'cpu',
        **kwargs
    ):
        actor_hparams = {
            **kwargs.get("actor_hparams", {})
        }
        critic_hparams = {
            **kwargs.get("critic_hparams", {})
        }
        encoder_hparams = {
            **kwargs.get("encoder_hparams", {})
        }
        tokenizer_hparams = {
            **kwargs.get("tokenizer_hparams", {})
        }

        self.device = device

        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path=encoder_name,
            **encoder_hparams
        )
        self.encoder.eval()
        self.encoder.to(self.device)
        embed_dims = self.encoder.config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_name,
            **tokenizer_hparams
        )
        vocab_size = self.tokenizer.vocab_size

        output_dims = output_dims or vocab_size

        ppo_kwargs = dict(
            actor_hparams=actor_hparams,
            critic_hparams=critic_hparams
        )

        if verbose:
            print(f"encoder ready on {self.device}. using {embed_dims} embedding dims")
            print(f"tokenizer using {vocab_size} different tokens")

        super(Agent, self).__init__(
            input_dims=embed_dims,
            output_dims=output_dims,
            gamma=gamma,
            epochs=epochs,
            gae_lambda=gae_lambda,
            policy_clip=policy_clip,
            entropy=entropy,
            batch_size=batch_size,
            actor_ridge=actor_ridge,
            actor_coefficient=actor_coefficient,
            actor_clip_grad_norm=actor_clip_grad_norm,
            critic_ridge=critic_ridge,
            critic_coefficient=critic_coefficient,
            critic_clip_grad_norm=critic_clip_grad_norm,
            name=name,
            verbose=verbose,
            load=load,
            chkpt_dir=chkpt_dir,
            device=device,
            **ppo_kwargs
        )


    def choose_action(
        self,
        state: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This is the basis of the idea of the alectors
        
        a) take the natural language description of the current state,
        b) encode/embed it to a vector space
        c) pass the embeddings to the actor/critic model
        d) return action
        """
        embeddings = self._embed(state)
        dist, value = self._sample_networks(embeddings)

        action = dist.sample()
        probs = dist.log_prob(action)

        return action, probs, value

    def learn(self) -> dict[str, list[float]]:
        """
        Main learning function for PPO agents.
        For details look up the paper in arxiv  
        """
        num_states = len(self._memory['states'])
        loss_history = dict(
            actor = [],
            critic = [],
            entropy = [],
            total = []
        )
        for _ in tqdm(range(self._epochs), disable=not self._verbose, desc="Learning Epoch", leave=False):
            advantage = self._calculate_advantage(num_states)

            for batch in self._batch_memory(num_states):
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                states = [self._memory['states'][i] for i in batch]
                embeddings = self._embed(states)

                losses = self._train_networks(embeddings, advantage, batch)
                for key, tensor in zip(loss_history.keys(), losses):
                    loss_history[key].append(tensor.item())

            torch.cuda.empty_cache()
        self._memory = self._set_memory()
        torch.cuda.empty_cache()
        return loss_history

    def _embed(self, state: str) -> torch.Tensor:
        encoded_state = self.tokenizer(
            state,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        encoder_output = self.encoder(**encoded_state)
        embeddings = encoder_output['last_hidden_state']
        return embeddings
