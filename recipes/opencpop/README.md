# opencpop

Recipe of the [Opencpop corpus](https://wenet.org.cn/opencpop/).

## How to use

You must first agree to the Opencpop's lincense to download the database.
Please follow the official guide: https://wenet.org.cn/opencpop/download/.

After downloading the database, please run the following command to convert the corpus to NNSVS's structure:

```
./run.sh  --stage 0 --stop-stage 0 --db-root /your/path/to/opencpop/segments
```

Then, you can run recipes as the same as the other recipes.

e.g.,

Feature extraction
```
./run.sh  --stage 1 --stop-stage 1
```

## Demo samples

https://r9y9.github.io/projects/nnsvs/
