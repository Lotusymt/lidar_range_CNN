import torch
from model import PairRangeCNN   # make sure this matches your file name


def main():
    # 1) Create model and load weights
    model = PairRangeCNN()
    state = torch.load("pair_range_cnn_kitti_00_10.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()  # inference mode (disables dropout, etc.)

    # 2) Example input for tracing: (B=1, C=2, H, W)
    H, W = 64, 1024  # <-- put your real range image size here
    example_input = torch.randn(1, 2, H, W)

    # 3) Trace and save TorchScript model
    traced = torch.jit.trace(model, example_input)
    traced.save("pair_range_cnn_kitti_00_10.pt")
    print("Saved TorchScript model to pair_range_cnn_kitti_00_10.pt")


if __name__ == "__main__":
    main()
