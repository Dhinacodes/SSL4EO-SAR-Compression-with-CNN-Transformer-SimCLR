class SimCLRSARDataset(Dataset):
    def __init__(self, root_dir, folder_limit=None):
        self.root_dir = root_dir
        self.samples = []
        zarr_folders = sorted(os.listdir(root_dir))[:folder_limit]

        for folder in zarr_folders:
            folder_path = os.path.join(root_dir, folder)
            if not folder_path.endswith(".zarr"):
                continue
            z = zarr.open(folder_path, mode='r')
            bands = z['bands']
            for i in range(bands.shape[0]):
                self.samples.append((folder_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        zarr_path, sample_idx = self.samples[idx]
        z = zarr.open(zarr_path, mode='r')
        sar_cube = z['bands'][sample_idx]  # [T=4, 2, H, W]

        processed = []
        for t in range(sar_cube.shape[0]):
            vv = sar_cube[t, 0]
            vh = sar_cube[t, 1]

            vv_denoised = bilateral_denoise(vv)
            vh_denoised = bilateral_denoise(vh)
            vv_clahe = apply_clahe(vv_denoised)
            edge_map = sobel_edges(vv_clahe)

            processed.append(np.stack([vv_denoised, vh_denoised, vv_clahe, edge_map], axis=0))

        processed = np.stack(processed)  # [T=4, 4, H, W]
        mean_img = np.mean(processed, axis=0)
        std_img = np.std(processed, axis=0)
        diff_img = processed[-1] - processed[0]

        temporal_stack = np.concatenate([mean_img, std_img, diff_img], axis=0)
        downsampled = np.stack([downsample(temporal_stack[c]) for c in range(temporal_stack.shape[0])])

        return torch.tensor(downsampled, dtype=torch.float32), torch.tensor(downsampled, dtype=torch.float32)
