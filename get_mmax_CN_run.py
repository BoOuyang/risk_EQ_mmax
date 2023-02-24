def generate_cascades(source_file, output_dir, nround, Delta, muD, delta, region = [60,150,0,60]):
    # import libraries
    import geopandas as gpd
    import pandas as pd
    import get_mmax_CN_method as gmcm
    import os

    # read source data
    source = gpd.read_file(source_file)

    # set initial number of fault segments
    nfltSS = source.shape[0]

    # change working directory to output directory
    os.chdir(output_dir)

    # main loop: generate cascades
    n = nfltSS
    ntot = 0  # total number of cascades generated

    for round in range(nround):
        id_new = round * 10000  # each new round of cascades gets new set of IDs

        # read cascade .shp file
        if(round > 0):
            cascade_name = './file_cascade_' + str(round-1) + '.shp'

            # break from loop if no more cascades gets new set of IDs
            if(not os.path.isfile(cascade_name)):
                break

            cascade = gpd.read_file(cascade_name)
            n = cascade.shape[0]  # number of cascading events for this round
            ntot = ntot + n  # number of cascades for the whole loop

        # loop through every fault segment in source
        for i in range(n):
            # define initial source (init) to propagate
            if(round == 0):
                init = source.loc[source['id'] == i]
                init.reset_index(drop=True, inplace=True)
            elif(round > 0):
                init = cascade.loc[pd.to_numeric(cascade['id'], errors='coerce').astype('Int64') == (i + (round-1) * 10000)]
                init.reset_index(drop=True, inplace=True)

            # 1. jumping
            res1 = gmcm.jump(source, init, region, Delta, round)

            # 2. bending/branching
            if(res1.shape[0] > 0):
                res2 = gmcm.bendbranch(source, init, region, res1, muD, delta)

                # 3.rupture progation
                if(res2.shape[0] > 0):
                    id_new = gmcm.propa(source, init, region, res2, id_new, round)
