import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image
from utils.utils import ImageDataset 
from utils.models import VGGEncoder, Decoder
import torch.optim as optim 
from tqdm import tqdm
from utils.utils import get_transform
from utils.utils import adaptive_instance_normalization, calc_mean_std

def parse_arguments():
    parser = argparse.ArgumentParser()
    # Add your arguments here
    parser.add_argument("--content-dir", type = str , default='C:/Users/DEVENDRA/OneDrive/Desktop/major projects/NST_PROJECT/content_data/test2017', help="Path to the content images directory")
    parser.add_argument("--style-dir", type = str , default='C:/Users/DEVENDRA/OneDrive/Desktop/major projects/NST_PROJECT/style_data', help="Path to the style images directory")
    parser.add_argument("--vgg", type = str , default='vgg_normalised.pth', help="Path to pre-trained VGG weights")
    parser.add_argument("--experiment", type = str , default='experiments2', help="Name of the experiment")
    parser.add_argument("--final_size", type=int, default=256, help="Final size of the images after resizing")
    parser.add_argument("--content_Size", type=int, default=512, help="Size of the content images after resizing")
    parser.add_argument("--style_Size", type=int, default=512, help="Size of the style images after resizing")
    parser.add_argument("--crop" ,action='store_true',default=True, help="Whether to crop the images after resizing")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning Rate')
    parser.add_argument("--lr_decay", type=float,default=5e-5 , help='Learning Rate Decay')
    parser.add_argument('--epochs', type=int , default=10, help='Number of epochs')
    parser.add_argument("--content_weight", type=float,default=1.0, help='Content weight')
    parser.add_argument("--style-weight", type=float, default=10, help='style weight')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=2, help='Save interval')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training')
    parser.add_argument('--decoder_path', type=str, default=None, help='Path to decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, default=None, help='Path to optimizer checkpoint')
    return parser.parse_args()




def main():
    args = parse_arguments()

    # Set device to GPU if available, otherwise use CPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path('experiment')/args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arg values
    with open(save_dir / "args.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    #  Create datasets and dataloaders
    
    content_transform =get_transform(args.content_Size, args.crop,args.final_size)
    style_transform =get_transform(args.style_Size, args.crop,args.final_size)

    content_dataset = ImageDataset(args.content_dir, content_transform)    
    style_dataset = ImageDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=False,drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=False,drop_last=True)

    print(f"Content dataset size: {len(content_dataloader)}")
    print(f"Style dataset size: {len(style_dataloader)}")

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(decoder.parameters(),lr= args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )
    
    print("Training Started.....")

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    running_loss = None
    running_closs = None
    running_sloss = None

    for epoch in range(args.epochs):
        progress_bar = tqdm(zip(content_dataloader ,style_dataloader), 
                            total = min(len(content_dataloader), len(style_dataloader)))


        running_loss = 0
        running_closs = 0
        running_sloss = 0
        
        for content_batch ,style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # print(len(c_feats))
            # print(len(s_feats))

            # print(c_feats[0].shape)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            # content loss
            loss_c = mse_loss(g_feats[-1], t) * args.content_weight 

            # ṣtyle loss
            loss_s = 0 
            for g_f , s_f in zip(g_feats,s_feats):
                g_mean , g_std = calc_mean_std(g_f)
                s_mean , s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean , s_mean) + mse_loss(g_std , s_std)

            loss_s = loss_s * args.style_weight 

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f'Loss:{loss.item():4f} , content_loss:{loss_c.item():4f},Style_loss:{loss_s.item():4f}')


            running_loss += loss.item()
            running_closs +=loss_c.item()
            running_sloss +=loss_s.item()

        scheduler.step()

        running_loss /= len(content_dataloader) 
        running_closs /= len(content_dataloader) 
        running_sloss /= len(content_dataloader)

        if (epoch +1) % args.log_interval ==0:
            tqdm.write(f"Iter {epoch+1} : Loss:{running_loss:.4f}, content_loss:{running_closs:.4f}, style_loss:{running_sloss:.4f}")

        if (epoch+1) % args.save_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch+1}.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch+1}.pth')

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch+1}.png', nrow=args.batch_size)


if __name__ == "__main__":
    main()

