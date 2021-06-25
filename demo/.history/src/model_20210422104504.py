
import numpy as np
import torch
import torch.nn as nn
import torchvision

# model class
class VGG16_HPMI(torch.nn.Module):

  # init function
  def __init__(self, model, num_classes=2):
    super().__init__()

    # pool layer
    self.pool = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=2))

    # spatial attention
    self.spatial_attention = torch.nn.Sequential(
        torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1),
        torch.nn.BatchNorm2d(1),
        torch.nn.Sigmoid()
    )

    # spatial attention
    self.spatial_attention2 = torch.nn.Sequential(
        torch.nn.Conv2d(64, 1, kernel_size=7, padding=3, stride=1),
        torch.nn.BatchNorm2d(1),
        torch.nn.Sigmoid()
    )
    # spatial attention
    self.spatial_attention3 = torch.nn.Sequential(
        torch.nn.Conv2d(128, 1, kernel_size=7, padding=3, stride=1),
        torch.nn.BatchNorm2d(1),
        torch.nn.Sigmoid()
    )
    # spatial attention
    self.spatial_attention4 = torch.nn.Sequential(
        torch.nn.Conv2d(256, 1, kernel_size=7, padding=3, stride=1),
        torch.nn.BatchNorm2d(1),
        torch.nn.Sigmoid()
    )
    # spatial attention
    self.spatial_attention5 = torch.nn.Sequential(
        torch.nn.Conv2d(512, 1, kernel_size=7, padding=3, stride=1),
        torch.nn.BatchNorm2d(1),
        torch.nn.Sigmoid()
    )

    # channel attention
    self.max_pool_1 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=224, stride=224))
    self.max_pool_2 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=112, stride=112))
    self.max_pool_3 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=56, stride=56))
    self.max_pool_4 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=28, stride=28))
    self.max_pool_5 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=14, stride=14))
    self.avg_pool_1 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=224, stride=224))
    self.avg_pool_2 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=112, stride=112))
    self.avg_pool_3 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=56, stride=56))
    self.avg_pool_4 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=28, stride=28))
    self.avg_pool_5 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=14, stride=14))

    # features
    self.features_1 = torch.nn.Sequential(*list(model.features.children())[:3])
    self.features_2 = torch.nn.Sequential(*list(model.features.children())[3:6])
    self.features_3 = torch.nn.Sequential(*list(model.features.children())[7:10])
    self.features_4 = torch.nn.Sequential(*list(model.features.children())[10:13])
    self.features_5 = torch.nn.Sequential(*list(model.features.children())[14:17])
    self.features_6 = torch.nn.Sequential(*list(model.features.children())[17:20])
    self.features_7 = torch.nn.Sequential(*list(model.features.children())[20:23])
    self.features_8 = torch.nn.Sequential(*list(model.features.children())[24:27])
    self.features_9 = torch.nn.Sequential(*list(model.features.children())[27:30])
    self.features_10 = torch.nn.Sequential(*list(model.features.children())[30:33])
    self.features_11 = torch.nn.Sequential(*list(model.features.children())[34:37])
    self.features_12 = torch.nn.Sequential(*list(model.features.children())[37:40])
    self.features_13 = torch.nn.Sequential(*list(model.features.children())[40:43])

    self.avgpool = nn.AdaptiveAvgPool2d(7)

    # classifier
    self.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 7)
    )


  # forward
  def forward(self, x):
    x = self.features_1(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention2(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2


    x = self.features_2(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention2(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    x = self.pool(x)
    

    x = self.features_3(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention3(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_4(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention3(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.pool(x)

    x = self.features_5(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention4(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_6(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention4(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_7(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention4(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    
    x = self.pool(x)

    x = self.features_8(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_9(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_10(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    
    x = self.pool(x)

    x = self.features_11(x)
   # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_5(x) + self.avg_pool_5(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_12(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_5(x) + self.avg_pool_5(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    

    x = self.features_13(x)
    # CA
    scale_CA1 = torch.nn.functional.sigmoid(self.max_pool_5(x) + self.avg_pool_5(x)).expand_as(x)
    x1 = x * scale_CA1
    scale_CA2 = torch.cat((torch.max(x1, 1)[0].unsqueeze(1), torch.mean(x1, 1).unsqueeze(1)), dim=1)
    # SA
    scale_SA1 = self.spatial_attention(scale_CA2)
    x1 =x1 * scale_SA1
    
    # SA
    scale_SA2 = self.spatial_attention5(x)
    x2 =scale_SA2 * scale_CA1
    x= x1*x2
    
    x = self.pool(x)

    x = self.avgpool(x)
    x = x.view(x.shape[0], -1)
    x = self.classifier(x)
    return x

def CNN3(input_shape=(48, 48, 1), n_classes=8):
    """
    :param input_shape:
    :param n_classes:
    :return:
    """
    # input
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


