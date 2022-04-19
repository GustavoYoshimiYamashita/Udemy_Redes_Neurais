from pybrain3.tools.shortcuts import buildNetwork
from pybrain3.datasets import SupervisedDataSet
from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.structure.modules import SoftmaxLayer
from pybrain3.structure.modules import SigmoidLayer

#rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer,
#                    hiddenclass = SigmoidLayer, bias = False)

rede = buildNetwork(2, 3, 1)
base = SupervisedDataSet(2, 1)
base.addSample((0, 0), (0, ))
base.addSample((0, 1), (1, ))
base.addSample((1, 0), (1, ))
base.addSample((1, 1), (0, ))

treianmento = BackpropTrainer(rede, dataset = base, learningrate= 0.01,
                              momentum = 0.06)

for i in range(1, 10000):
    erro = treianmento.train()
    if i % 1000 == 0:
        print("Erro: %s" % erro)

print(rede.activate([0,0]))
print(rede.activate([1,0]))
print(rede.activate([0,1]))
print(rede.activate([1,1]))
