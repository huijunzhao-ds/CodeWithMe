import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy


def image_loader(image_name, transforms):
    image = Image.open(image_name)
    image = transforms(image).unsqueeze(0)
    return image.to(torch.float)


def image_unloader(tensor):
    image = tensor.cpu().clone().squeeze(0)   # we clone the tensor to not do changes on it
    image = transforms.ToPILImage()(image)
    return image


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def gram_matrix(x):
    a, b, c, d = x.size()  
    features = x.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std, device):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).to(device).view(-1, 1, 1)
        self.std = torch.tensor(std).to(device).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
   

def get_style_model_and_losses(cnn, style_img, content_img, device, 
                               normalization_mean=[0.485, 0.456, 0.406], # para vgg is trained on
                               normalization_std=[0.229, 0.224, 0.225], # para vgg is trained on
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)
    cnn.to(device)
    normalization = Normalization(normalization_mean, normalization_std, device)
    model = nn.Sequential(normalization)
    content_losses,  style_losses = [], []

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

def run_style_transfer(cnn, content_img, style_img, input_img, device, 
                       max_steps=500, style_weight=1000000, content_weight=1, tol=1e-4):
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, device)
    optimizer = get_input_optimizer(input_img)

    iteration = [0]
    prev_loss = torch.tensor([float("Inf")])
    while iteration[0] <= max_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score, content_score = 0, 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            curr_loss = style_weight * style_score + content_weight * content_score
            curr_loss.backward()

            iteration[0] += 1
            if iteration[0] % 25 == 0:
                print("iteration {}:".format(iteration))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_weight * style_score.item(), content_weight * content_score.item()))
                print()

            if prev_loss[0] - curr_loss < tol:
                return curr_loss
            else:
                prev_loss[0] = curr_loss
            return curr_loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

if __name__=="__main__":
    cnn = models.vgg19_bn(pretrained=True,progress=False).features.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 512 #if torch.cuda.is_available() else 128 
    loader = transforms.Compose([
        transforms.Resize(img_size),  # scale imported image
        transforms.ToTensor()])

    style_img = image_loader("style.png", loader)
    content_img = image_loader("content.png", loader) 
    input_img = content_img.clone()
    
    output = run_style_transfer(cnn, content_img, style_img, input_img, device, max_steps=10)
    image_unloader(output).save("result.png")