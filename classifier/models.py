import os

from model_trainer.models import Trainer

from data.constants import saved_sessions_root

from .helpers import show_stats


class Classifier:

    def __init__(self, model):
        self.model = model
        self.trainer = Trainer()

    @property
    def name(self):
        return self.model.name

    @property
    def session_file(self):
        return os.path.join(saved_sessions_root, '{}.session'.format(self.name))

    def has_existing_session(self):
        return os.path.exists('{}/{}.session'.format(saved_sessions_root, self.name))

    def train(self, data_for_trainer, **kwargs):
        data_for_trainer.print_shapes()
        self.trainer.set_training_hyper_parameters(self.model.hyper_parameters)
        self.trainer.run(self.model, data_for_trainer, **self._get_training_options(**kwargs))

    def _get_training_options(self, **kwargs):
        all_batches = kwargs.get('all_batches', False)
        save = kwargs.get('save', False)
        from_disk = False

        if kwargs.get('from_disk', False) and self.has_existing_session():
            from_disk = True

        options = {
            'all_batches': all_batches,
            'save': save,
            'from_disk': from_disk,
            'session_file': self.session_file,
        }
        return options

    def stats(self):
        show_stats(self.trainer.steps, self.trainer.losses, self.trainer.accuracies)
