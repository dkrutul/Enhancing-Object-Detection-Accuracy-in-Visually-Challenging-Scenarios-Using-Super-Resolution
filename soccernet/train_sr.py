import cv2
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision as tv
from core.models import RLFN_S
from core.utils import tensor2uint
from torch.nn.functional import l1_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

dataset_name = "4k_UHDSR8K"
data_path = f"./data_{dataset_name}/"


MAX_EPOCHS = 15000
batch_size = 32

LR = 0.01
def lr_lambda(epoch):
    return max(0.999 ** epoch, LR) 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for upscale in [2, 3, 4, 6]:
    model = RLFN_S(
        in_channels=3, out_channels=3,
        upscale=upscale
    )

    # Choose optimizer
    opt =  torch.optim.Adam(model.parameters(), LR)
    scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Set input and output dimientions
    LR_dim = (128, 128)
    HR_dim = (int(LR_dim[0]*upscale), int(LR_dim[1]*upscale))

    resize_obj = tv.transforms.Resize(LR_dim)

    transform = transforms.Compose([
        transforms.RandomCrop(HR_dim, pad_if_needed =True),
        transforms.ToTensor()
    ])

    dataset = tv.datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

    
    all_epochs = MAX_EPOCHS
    for epoch in tqdm(range(all_epochs)):
        #Load images from directory
        images, labels = next(iter(dataloader))
        #Move model and dataset from CPU to GPU
        gpu_model = model.to(device)
        images = images.to(device)

        HR_crop = images
        LR_crop = resize_obj(HR_crop)

        # Push images to model
        SR_img = gpu_model(LR_crop)

        #Calculate loss value and do backward poropagation
        t = l1_loss(SR_img, HR_crop)
        t.backward()
        opt.step()
        opt.zero_grad()
        # Step the scheduler
        scheduler.step()

        if epoch % 1000 == 0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, dpi=150, figsize=(16,16))

            print("Epoch:" + str(epoch) + "/" + str(all_epochs) +  " output loss:" + str(float(t)))
            SR_img = tensor2uint(SR_img[0])
            HR_img = tensor2uint(HR_crop[0])
            LR_img = tensor2uint(LR_crop[0])

            ax1.imshow(LR_img)
            ax2.imshow(HR_img)
            ax3.imshow(SR_img)
            ax4.imshow(cv2.resize(LR_img, HR_dim, cv2.INTER_CUBIC))
            plt.savefig(f"model/checkpoints/checkpoint_x{upscale}_{epoch}_{batch_size}_{dataset_name}.svg")
            plt.show()
            torch.save(model.state_dict(), f"model/checkpoints/checkpoint_x{upscale}_{epoch}_{batch_size}_{LR}_{dataset_name}.pth")            
            
    
    torch.save(model.state_dict(), f"model/save_x{upscale}_{MAX_EPOCHS}_{batch_size}_{LR}_{dataset_name}_LRdim{LR_dim[0]}.pth")
