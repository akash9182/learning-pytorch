import torch 
import torchvision 
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

#======================= Basic autograd example 1 =======================#
# Create tensors.
x = Variable(torch.Tensor([1]), requires_grad = True)
w = Variable(torch.Tensor([2]), requires_grad = True)
b = Variable(torch.Tensor([3]), requires_grad = True)

# Build a computational graph.
y  = w * x + b

#Compute gradients
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)



#======================== Basic autograd example 2 =======================#
# Create tensors.
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

linear = nn.Linear(3, 2)
print('w:', linear.weight)
print('b:', linear.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr = 0.01)

pred = linear(x)

loss = criterion(pred, y)
print('loss:' , loss.data[0])

loss.backward()

print('dl/dw:', linear.weight.grad)
print('dl/db:', linear.bias.grad)

optimizer.step()

print('loss after 1 step optimization: ', loss.data[0])
pred = linear(x)
loss = criterion(pred, y)

#======================== Loading data from numpy ========================#
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a)   # convert numpy array to torch tensor
c = b.numpy()			# convert torch tensor to numpy array

#========================= Implementing the input pipline=================#

train_datadset = dsets.CIFAR10(root='../data/', 
								train=True,
								transforms=transforms.ToTensor(),
								download=True)

#select one data pair 
image, label = train_datadset[0]
print(image.size())
print(label)

train_loader = torch.utils.data.DataLoader(dataset=train_datadset,
											batch_size=100,
											shuffle=True,
											num_workers=2)

data_iter =	iter(train_loader)

images,labels = data_iter.next()

for images, labels in train_loader:
	# your training code 
	pass

#=================Input pipeline for custom dataset===================#

#custom  dataset as below
class CustomDataset(data.Dataset):
	def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names
		pass

	def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
		pass

	def __len__(self):
        # You should change 0 to the total size of your dataset.
		return 0

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
											batch_size=100,
											shuffle=True,
											num_workers=2)


#========================== Using pretrained model ==========================#
# Download and load pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

for parm in resnet.parameters():
	param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# For test.
images = Variable(torch.randn(10, 3, 256, 256))
outputs = resnet(images)
print (outputs.size())   # (10, 100)


#============================ Save and load the model ============================#
# Save and load the entire model.
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# Save and load only the model parameters(recommended).
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))