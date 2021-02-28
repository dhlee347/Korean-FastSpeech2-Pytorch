from sklearn.preprocessing import StandardScaler

def build_from_path(in_dir, out_dir, meta):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    train, val = list(), list()

    # scalers for mel, f0, energy
    scalers = [StandardScaler(copy=False) for _ in range(3)]

    n_frames = 0
    with (in_dir / meta).open(encoding='utf-8') as f:
        for index, line in enumerate(f):

            parts = line.strip().split('|')
            basename, text = parts[0], parts[3]

            ret = process_utterance(in_dir, out_dir, basename, scalers)

            if ret is None:
                continue
            else:
                info, n = ret
            
            if basename[0] == '1':
                val.append(info)
            else:
                train.append(info)

            if index % 100 == 0:
                print("Done %d" % index)

            n_frames += n

    param_list = [np.array([scaler.mean_, scaler.scale_]) for scaler in scalers]
    param_name_list = ['mel_stat.npy', 'f0_stat.npy', 'energy_stat.npy']
    [np.save(os.path.join(out_dir, param_name), param_list[idx]) for idx, param_name in enumerate(param_name_list)]

    return [r for r in train if r is not None], [r for r in val if r is not None]
