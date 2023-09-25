echo "Start training Target model: VGG_16 ......"
python train_classifier.py
echo "Start training Improved GAN model: Diffusion-KED ......"
python run_k+1_gan.py
echo "Start training Legacy GAN model: GMI ......"
python run_binary_gan.py
echo "Start doing Inversion attack and recovery the privacy pictures ......"
python recovery.py
echo "Finish! ......"