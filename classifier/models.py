import os

from model_trainer.models import ModelTrainer

from data.constants import saved_sessions_root

from .helpers import show_stats


class Classifier:

    def __init__(self, model):
        self.model = model
        self.trainer = ModelTrainer()

    @property
    def name(self):
        return self.model.name

    @property
    def session_file(self):
        return os.path.join(saved_sessions_root, '{}.session'.format(self.name))

    def has_existing_session(self):
        return os.path.exists('{}/{}.session'.format(saved_sessions_root, self.name))

    def train(self, *args, **kwargs):
        save = kwargs.get('save', False)
        from_disk = False

        if kwargs.get('from_disk', False) and self.has_existing_session():
            from_disk = True

        self.trainer.run(self.model, *args, save=save, from_disk=from_disk,
                         session_file=self.session_file, **self.model.hyper_parameters)

    def stats(self):
        show_stats(self.trainer.steps, self.trainer.losses, self.trainer.accuracies)
