from avalanche.benchmarks import SplitMNIST
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics

scenario = SplitMNIST(n_experiences=5)
model = MyModel()  # เช่น BERT-based classifier
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

evaluator = EvaluationPlugin(
    accuracy_metrics(experience=True, stream=True),
)
strategy = Naive(model, optimizer, loss=nn.CrossEntropyLoss(), train_mb_size=32, train_epochs=1, evaluator=evaluator)

for experience in scenario.train_stream:
    strategy.train(experience)
    strategy.eval(scenario.test_stream)
