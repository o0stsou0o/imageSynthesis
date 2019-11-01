# imageSynthesis
This is an initial attempt to generate stimuli to drive neural populations in macacque v4 beyond maximum firing rates based on the image dataset Bold5000 (Chang, Pyles, Gupta, Tarr, Aminoff et. al.).  The data was collected by Ruobing Xia and David Sheinberg with microelectrode data with 217 channels with 4899 stimuli.  

For each channel, there were only 2 or 3 repeats of each stimuli thus making this very noisy data for this experiment and making this experiment more of a trial run.  

The process of driving neural populations was inspired by Neural Population Control Via Deep Image Synthesis (Bashivan, Kar, Dicarlo) and Evolving Super Stimuli For Real Neurons Using Deep Generative Networks (Ponce, Xiao, Schade, Hartmann, Kreimann, Livingstone).  The time window of recording for each stimuli was 915ms.  We averaged the middle 305 ms as a single datapoint and we averaged across the repeats of each stimuli.  

We then took a pre-trained VGG-16 and fed the Bold5000 stimuli into the network to predict the response of each of the 217 stimuli.  Again by response, we mean the average across the middle 305ms averaged across the repeats of the specific stimuli.  We performed transfer learning after MaxPool 4 and another transfer learning after MaxPool3 of the VGG-16 with a fully connected layer.  We froze the weights of the VGG network and trained the new fully connected layer but used lasso regularization to make the layer sparse.  

We trained for 100 epochs with the transfer after both the MaxPool3 and MaxPool4.  After these 100 epochs, the pearson correlation between ground truth and predictions across all stimuli and neurons is over .6 for both layer transfers.  

We then used a network inspired by Inception In Visual Cortex: In Vivo Silico Loops Reveal Most Exciting Images (Walke, Sinz, Ecker, Tolias, et al.) to go backwards into the now trained VGG-16 networks and generate stimuli that drive the neurons beyond the previous maximum.   





