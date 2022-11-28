from args import get_args, modify_args, check_args
from dataset import CustomDataset
from train import PytorchTrainer
import config
from network import Network
from utils import set_seed

args = get_args()
args = modify_args(args)
args = check_args(args)
config.args = args

set_seed(args.random_state)

dataset = CustomDataset()
dataset.download()
dataset.create()


exit()

model = Network()

trainer = PytorchTrainer(dataset, model)
trainer.train()
trainer.save()



