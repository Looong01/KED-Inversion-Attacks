Write-Host "Start training Target model: VGG_16 ......"
python train_classifier.py
Write-Host "Start training Improved GAN model: Diffusion-KED ......"
python run_k+1_gan.py
Write-Host "Start training Legacy GAN model: GMI ......"
python run_binary_gan.py
Write-Host "Start doing Inversion attack and recovery the privacy pictures ......"
python recovery.py
Write-Host "Finish! ......"