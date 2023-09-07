from torchvision.transforms import transforms, Resize, ToTensor, AutoAugment, AutoAugmentPolicy


class Config:
    batch_size = 64
    learing_rate = 1e-5
    epochs = 10

    def __init__(self):
        pass


class GlomeruliTrainConfig(Config):
    batch_size = 8
    learning_rate = 1e-5
    epochs = 2
    momentum = 0.9
    num_classes = 2
    aug = False
    train = True
    transform = transforms.Compose([
        ToTensor()
    ])


class GlomeruliTrainConfig3(GlomeruliTrainConfig):
    num_classes = 3


class GlomeruliTrainConfigViT(Config):
    batch_size = 8
    learning_rate = 1e-5
    epochs = 50
    num_classes = 2
    aug = False
    train = True
    transform = transforms.Compose([
        Resize(224),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        ToTensor()
    ])


class GlomeruliTestConfigViT(GlomeruliTrainConfigViT):
    batch_size = 64
    train = False
    transform = transforms.Compose([
        Resize(224),
        ToTensor()
    ])


class GlomeruliTestConfigViT3(GlomeruliTestConfigViT):
    num_classes = 3


class GlomeruliTestConfig(GlomeruliTrainConfig):
    batch_size = 64
    train = False


class GlomeruliTestConfig3(GlomeruliTrainConfig3):
    batch_size = 64
    train = False
