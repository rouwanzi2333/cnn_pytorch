import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 数据集准备，这里使用的数据集是mnist数据集，具体数据集详情可见以下链接：https://blog.csdn.net/bwqiang/article/details/110203835
# 我提取到的关于数据集的信息关键点：数据全量：70000（60000张训练图片和10000张测试图片）；分类数：10(分别代表0-9)；图片的尺寸为 28*28；黑白图像
# mnist数据集已内置到pytorch中，可以直接调用
# step：加载数据集
# pytorch.torchvision.transform可以对PIL.Image进行各种变换
# torchvision.transforms.ToTensor：把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
# Normalize：用均值和标准差归一化张量图像；归一化，这里的0。1307、0。3081分别表示对张量进行归一化的全局平均值和方差，因为图像是灰色的只有一个通道，
# 所以分别指定一了一个值，如果有多个通道，需要有多个数字，如3个通道，就应该是Normalize([m1, m2, m3], [n1, n2, n3])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])
# 导入训练数据集，root ：需要下载至地址的根目录位置；train=True：代表导入训练数据集，download=True：是否下载到root指定的位置，如果指定的root位置已经存在该数据集，则不再下载，可以看到自己的本地目录里多了dataset，里面有各种数据文件，
# transform：一系列作用在PIL图片上的转换操作，返回一个转换后的版本
#batch_size：设置每次传入的数据量；shuffle=True：是否打乱数据集
#为嘛要用Dataloader：数据要按照batchsize大小慢慢传入训练
batch_size = 100
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#导入测试数据集
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

#构建网络，网络定义可查看：https://blog.csdn.net/WRWEREWRET/article/details/118752656
class ConvolutionNet(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNet,self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 5) #第一个卷积层
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size = 5) #第二个卷积层
        self.pooling = torch.nn.MaxPool2d(2) #最大池化，步数为2
        self.fc = torch.nn.Linear(320, 10) #最后输出10个概率值，取最大的那个

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x))) #ReLU (Rectified Linear Unit) 是一种常用的激活函数，它的作用是将输入值限制在非负范围内，并且在正半轴上具有线性性质
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) #拉直，作为后面全连接层torch.nn.Linear的输入
        x = self.fc(x)
        return x

model = ConvolutionNet()
#构建损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

#构建训练和测试函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0): #enumerate函数的作用：迭代循环，batch_idx为从0开始的索引值, data为具体取数，
        #详细可看：https://blog.csdn.net/weixin_44025103/article/details/124911947
        inputs, target = data #inputs为data[batch_idx]中的特征列，target为标签列
        #print(inputs.shape) #输出：torch.Size([10, 1, 28, 28]) 样本量为10，通道数1；28*28
        #print(target) #输出：tensor([6, 7, 9, 6, 6, 9, 6, 5, 4, 2]) 标签类，10个类别，为什么是10个，因为batchsize是10，取了10条数据，所以有10个y
        optimizer.zero_grad() #清空过往梯度

        outputs = model(inputs) #模型的输出
        loss = criterion(outputs, target) #计算输出与原始标签列的损失
        loss.backward() #反向传播，计算当前梯度
        optimizer.step() #更新网络参数

        running_loss += loss.item() #叠加损失running_loss=running_loss+loss
        if batch_idx % 300 == 299: #%表示整除，这句话意思是当batch_idx为300的倍数时，print下面的信息
            # print，每个epoch，对应的batch_idx，此时的损失是多少，目的是为了观察损失，因为损失足够小时，可以停止训练了，防止过拟合
            print('[%d, %5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0

def test(): #部分函数用法同train
    correct = 0 #用于存放预测正确的个数
    total = 0 #用于存放全部个数
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            outputs = model(inputs) #将特征的那些列作为模型的输入
            #_, x这是因为torch.max(a,dim=b)会返回两个值，一是最大值的索引，即哪一个数最大，二是返回数值，即这个最大的数等于几。
            # _，是占位符，表示我们知道这里有一个数，但是用不到它。
            #目的：取出output，即torch.nn.Linear(320, 10)输出的10个概率中，最大的那一个，就是最后的预测结果
            _, predicted = torch.max(outputs.data, dim = 1)
            #print("predicted",predicted) #输出：tensor([2, 6, 5, 0, 1, 2, 3, 4, 5, 6]) 10个预测结果
            total += target.size(0)
            #print(target.size(0)) #输出：10
            correct += (predicted == target).sum().item() #判断predicted预测得到的结果是否和标签一样
    print('Accuracy on test set:%d %% [%d/%d]' % (100 * correct / total, correct, total)) #计算正确率
    return 100 * correct / total #返回正确率

if __name__ == '__main__':
    len =1000 #epoch大小
    x = [0] * len
    y = [0] * len
    for epoch in range(10): #按照epoch=10进行循环
        train(epoch) #对每个epoch调用train函数
        y[epoch] = test() #调用test函数
        x[epoch] = epoch #画图横坐标用
    plt.plot(x, y) #最后可以得到每个epoch对应的准确率
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
#hot-fix

