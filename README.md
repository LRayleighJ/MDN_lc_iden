# Mixture Density Network identifying binary lensing events
## File structure

I tried to write something clear but failed. About the main idea of this project, please read the pdf files in the folder `/science/`. 

* `dataprocessing`: doing some dataset process
* `network`: testing MDN
    
    * `netmodule`: main structure of network
    
    * `datamodule`: some functions for calculating and data processing
* `GRU`: testing RNN+ResNet+MDN
* `simudata`: generating simulation data
* `testnet`: testnet

If having some questions, please contact liuzrua@163.com, though he doesn't know how to deal these problems either.

## Simulate data generation

First, you should download `noisetimedata.zip` and unzip it to your local folder(`$noisetimedata`). `noisetimedata.zip` contains 2 folders,`/timeseq` and `/noisedata_hist`. Download URL is [https://cloud.tsinghua.edu.cn/f/c2653e17fed945229d20/?dl=1](https://cloud.tsinghua.edu.cn/f/c2653e17fed945229d20/?dl=1)

The simulation code can be seen in `/datagenerate/generate1.py`. 2 Python Class `TimeData` and `NoideData` are defined.

`Class TimeData`:

initialization parameters `(datadir, num_point, bad_seq_threshold=35)`:

* `datadir`: `$noisetimedata/timeseq/`
* `num_point`: The number of time points you want.
* `bad_seq_threshold`: A parameter controls the maxium cadence of your time sequence. The maxium cadence would not larger than `bad_seq_cadence*(range_of_timeseq/num_point)`

functions:

* `TimeData.get_t_seq(max_count=20)` 

    Returns:

    * `time_cut`: Time sequences. We cut a piece of required nmumber of points from KMTNet data, and renormalize it by $t_n -> t_n-mean(t)$
    * `d_times`: The time cadence of each point. $\Delta t_n = \frac{t_{n+1}-t_{n-1}}{2}$

`Class NoiseData`:

initialization parameters: `(datadir)`:

* `datadir`: `$noisetimedata/noisedata_hist/`

functions:

* `noisemodel(mag)`:

    Input:

    * `mag`(ndarray,shape=(n,)): The magnitude data. The magnitudes should between 9.3 and 22.3.

    Returns:

    * `noise`: The gaussian noise corresponds to the input magnitude data.
    * `error`: The errorbar of each magnitude point. 