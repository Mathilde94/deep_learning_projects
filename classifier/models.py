import os
from sklearn.manifold import TSNE

from model_trainer.models import LSTMTrainer, Trainer, TextTrainer

from data.constants import saved_sessions_root

from .helpers import plot, show_stats


class BaseClassifier:
    def __init__(self, model):
        self.model = model
        self.trainer = self.trainer_class()

    @property
    def name(self):
        return self.model.name

    @property
    def session_file(self):
        return os.path.join(saved_sessions_root, '{}.session'.format(self.name))

    def has_existing_session(self):
        return os.path.exists('{}/{}.session'.format(saved_sessions_root, self.name))

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


class Classifier(BaseClassifier):
    trainer_class = Trainer

    def train(self, data_for_trainer, **kwargs):
        data_for_trainer.print_shapes()
        self.trainer.set_training_hyper_parameters(self.model.hyper_parameters)
        self.trainer.run(self.model, data_for_trainer, **self._get_training_options(**kwargs))


class TextClassifier(BaseClassifier):
    trainer_class = TextTrainer

    def train(self, data, reverse_dictionary, **kwargs):
        self.trainer.set_training_hyper_parameters({'reverse_dictionary': reverse_dictionary})
        self.trainer.set_training_hyper_parameters(self.model.hyper_parameters)
        self.final_embeddings = self.trainer.run(self.model, data, **self._get_training_options(**kwargs))

    def show_points(self):
        num_points = 400

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(self.final_embeddings[1:num_points + 1, :])
        words = [self.trainer.reverse_dictionary[i] for i in range(1, num_points + 1)]
        plot(two_d_embeddings, words)


class LSTMClassifier(BaseClassifier):
    trainer_class = LSTMTrainer

    def train(self, train_set, valid_set, **kwargs):
        self.trainer.set_training_hyper_parameters(self.model.hyper_parameters)
        self.trainer.run(self.model, train_set, valid_set, **self._get_training_options(**kwargs))
