import argparse


def main():
    parser = argparse.ArgumentParser(description='for dataset preprocessing')
    parser.add_argument('-s', '--split', help='test or train', default='train', type=str)
    parser.add_argument('-d', '--dataset_dir', help='dataset directory (vggface2 or lfw or UFPR or CASIA)', default='/data/maklemt/CASIA_WebFace', type=str) # /data/maklemt/VGGFace2
    args = parser.parse_args()

    if 'vggface2' in args.dataset_dir.lower():
        from data.VGGFace2_preprocessor import apply_preprocessing
        apply_preprocessing(args.split, args.dataset_dir, online_detection=False)
    elif 'lfw' in args.dataset_dir.lower():
        from data.LFW_preprocessor import apply_lfw_preprocessing
        apply_lfw_preprocessing(dataset_dir=args.dataset_dir)
    elif 'ufpr' in args.dataset_dir.lower():
        from data.UFPR_preprocessor import preprocess_ufpr
        preprocess_ufpr(args.dataset_dir)
    elif 'casia' in args.dataset_dir.lower():
        from data.CASIA_preprocessor import apply_preprocessing
        apply_preprocessing(dataset_dir=args.dataset_dir)
    else:
        print("Unknown dataset")


if __name__ == '__main__':
    main()
