import pandas as pd
import numpy as np
import pickle

class H2o_classifier():

    def __init__(self, max_depth):

        self.max_depth = max_depth
        self.model = None
        self.expert_identifier = None

    def start_server(self):
        import h2o

        h2o.init(max_mem_size='12G')

    def stop_server(self):
        import h2o

        h2o.cluster().shutdown()

    def fit(self, X, y, expert_identifier):
        import h2o
        from h2o.estimators.gbm import H2OGradientBoostingEstimator        

        self.expert_identifier = str(expert_identifier)

        df = pd.DataFrame(X)
        df_target = pd.DataFrame(y)    
        df["label"] = df_target
        frame = h2o.H2OFrame(df)

        # # assign target and inputs
        y = 'label'
        X = [name for name in frame.columns if name not in [y]]

        model_id = 'id_'.format(str(expert_identifier))

        # train single tree model_DT model
        model_DT = H2OGradientBoostingEstimator(ntrees=1,
                                                sample_rate=1,
                                                col_sample_rate=1,
                                                max_depth=self.max_depth,
                                                model_id=model_id)

        model_DT.train(x=X, y=y, training_frame=frame)

        # persist MOJO (compiled, representation of trained model from which to generate plot of model_DT
        mojo_path = model_DT.download_mojo(path='temp/model_DT_e{}_mojo.zip'.format(str(expert_identifier)))

        self.model = model_DT
        # Save model for plotting
        pickle.dump( model_DT, open( "temp/h2o_tree_e{}.p".format(str(expert_identifier)), "wb" ) )

        print(model_DT)
        print('Generated MOJO path:\n', mojo_path)


    def predict(self, X):
        import h2o

        frame = h2o.H2OFrame(X)
        #model_DT = pickle.load( open("temp/h2o_tree_e{}.p".format(str(expert_identifier)), "rb" ) )
        return self.model.predict(frame)["predict"]

    def plot(self):
        import os
        import re
        import subprocess
        from subprocess import CalledProcessError
        import time

        from h2o.backend import H2OLocalServer

        details = True # print more info on tree, details = True
        title = None

        expert_identifier = self.expert_identifier

        hs = H2OLocalServer()
        h2o_jar_path = hs._find_jar()
        print('Discovered H2O jar path:\n', h2o_jar_path)

        model_id = 'id_'.format(str(expert_identifier))
        mojo_path = "temp/model_DT_e{}_mojo.zip".format(str(expert_identifier))

        gv_file_name = 'temp/h2o_tree_e_' + self.expert_identifier + '.gv'
        gv_args = str('-cp ' + h2o_jar_path +
                    ' hex.genmodel.tools.PrintMojo --tree 0 -i '
                    + mojo_path + ' -o').split()
        gv_args.insert(0, 'java')
        gv_args.append(gv_file_name)

        if details:
            gv_args.append('--detail')

        if title is not None:
            gv_args = gv_args + ['--title', title]
            
        print()
        print('Calling external process ...')
        print(' '.join(gv_args))
            
        _ = subprocess.call(gv_args)        

        png_file_name = 'temp/h2o_tree_e_' + self.expert_identifier + '.png'
        png_args = str('dot -Tpng ' + gv_file_name + ' -o ' + png_file_name)
        png_args = png_args.split()

        print('Calling external process ...')
        print(' '.join(png_args))

        _ = subprocess.call(png_args, shell=True) 

