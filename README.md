# gsoc-atlas-anomaly
Two different models are use for this one is ml based isolation forest and the is an lstm autoencoder
The isolation forest model only uses a single feature the autoencoder uses multiple features 
Both of them are trained only for memory based anomalies but similar approach can be used for io

## data creation 
# isolation forest
In isolation forest a single run of the mem-burner was used for 4000 seconds(approx) at 100mb in which 0.5 of it was written

# lstm autoencoder
In autoencoder 5 different runs were made at different levels of memory usage to give the model idea about multiple possible normal values.

Data_none - baseline

Data_1 - low

Data_2 - medium

Data_3 - high

Data_4 - low
