import torch
from torchvision import transforms
import torchvision.transforms.functional as trf
import torch.utils.data as dutils
from typing import List
from PIL import Image
import os


class NoisyDataset(dutils.Dataset):
    """
    Indexing gives (ground truth, rendered) Tensor pairs.
    Cropping and flipping are implemented without torch.transforms, as we have to make sure the same transformation
    is applied to the gt and rendered image.
    """
    def __init__(self, root_path: str, iters: List[int] = None, crop_size=(512, 512), flipx=True, flipy=True,
                 transform=lambda x:x):
        def path_list(scene, test_or_train, iters):
            x = "gt" if iters is None else f"render_{iters}"
            path = os.path.join(root_path, scene, x, test_or_train)
            return [os.path.join(path, i) for i in os.listdir(path)]

        self.root_path = root_path
        self.iters = iters
        self.crop_size = crop_size
        self.flipx = flipx
        self.flipy = flipy
        self.transform = transform

        assert os.path.isdir(root_path)

        render_paths = []
        gt_paths = []

        # e.g. gt_paths = ["a", "b", "c"]
        #   render_paths = [["a1", "a2", "a3"], ...]

        # [a1, b1, c1]
        # [a2, b2, c2]

        for scene in os.listdir(root_path):
            gt_paths += path_list(scene, "train", None)
            gt_paths += path_list(scene, "test", None)

            rpaths = []
            for i in os.listdir(os.path.join(root_path, scene)):
                if i.startswith("render_"):
                    it = int(i[7:])
                    if iters is None or it in iters:
                        rpaths.append(path_list(scene, "train", it) + path_list(scene, "test", it))

            render_paths += list(zip(*rpaths))

        self.gt_paths = gt_paths
        self.render_paths = render_paths

    def __len__(self):
        return len(self.render_paths)

    def __getitem__(self, item):
        """Returns (gt, im) both of shape 3 x H x W where (H, W) = self.crop_size."""
        n = len(self.render_paths[item])
        rp = self.render_paths[item][torch.randint(0, n, ())]
        gp = self.gt_paths[item]

        tot = transforms.ToTensor()
        rim = tot(Image.open(rp))
        gim = tot(Image.open(gp))

        assert rim.shape == gim.shape

        if self.crop_size is not None:
            c, h, w = rim.shape

            # Deal with images which are below the crop size by scaling them up
            if h < self.crop_size[0] or w < self.crop_size[1]:
                scale = max(self.crop_size[0] / h, self.crop_size[1] / w) + 0.1
                new_h = int(scale * h)
                new_w = int(scale * w)

                rim = trf.resize(rim, size=[new_h, new_w])
                gim = trf.resize(gim, size=[new_h, new_w])

            # Perform the same random crop on both gt and rendered image
            i, j, h, w = transforms.RandomCrop.get_params(rim, output_size=self.crop_size)
            rim = trf.crop(rim, i, j, h, w)
            gim = trf.crop(gim, i, j, h, w)

        if self.flipx and torch.randint(0, 2, ()) == 1:
            rim = trf.hflip(rim)
            gim = trf.hflip(gim)

        if self.flipy and torch.randint(0, 2, ()) == 1:
            rim = trf.vflip(rim)
            gim = trf.vflip(gim)

        return self.transform(gim), self.transform(rim)






