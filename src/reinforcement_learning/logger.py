from stable_baselines3.common.callbacks import BaseCallback


class TrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.logs = {
            "value_loss": [],
            "entropy_loss": [],
            "policy_loss": [],
            "ep_rew_mean": [],
            "timesteps": [],
        }

    def _on_step(self) -> bool:
        # To będzie wywoływane co n_steps
        if "train/value_loss" in self.model.logger.name_to_value:
            self.logs["value_loss"].append(
                self.model.logger.name_to_value["train/value_loss"]
            )
            self.logs["entropy_loss"].append(
                self.model.logger.name_to_value["train/entropy_loss"]
            )
            self.logs["policy_loss"].append(
                self.model.logger.name_to_value["train/policy_gradient_loss"]
            )
            # self.logs["ep_rew_mean"].append(self.model.logger.name_to_value["rollout/ep_rew_mean"])
            if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
                self.logs["ep_rew_mean"].append(
                    self.model.logger.name_to_value["rollout/ep_rew_mean"]
                )
            if "rollout/ep_len_mean" in self.model.logger.name_to_value:
                self.logs["ep_len_mean"].append(
                    self.model.logger.name_to_value["rollout/ep_len_mean"]
                )
            self.logs["timesteps"].append(self.num_timesteps)
        return True
