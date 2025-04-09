def extract_embeddings(model, dataset, output_csv="sar_embeddings.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_embeddings = []

    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Extracting Embeddings"):
            x = x.to(device)
            embeddings = model(x).cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    df = pd.DataFrame(all_embeddings)
    df.to_csv(output_csv, index=False)
    print(f"📄 Embeddings saved to {output_csv}")
