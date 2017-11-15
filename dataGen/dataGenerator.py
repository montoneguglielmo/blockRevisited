class dataGeneratorPrototip(object):
    """An abstract class that returns the generator for the train validation and testset.
    """
    
    def __init__(self,**kwargs):
        pass

    
    def returnGen(self,sets,**kwargs):
       """
       This function should return in output the generator for the train, validation and test set
       Arguments:
          sets (string): it can have one of the three values: "test", "valid", "train"
       
       """
       pass


    def returnConfig(self, **kwargs):
        """
        Return a dictionary describing the dataset (for example the name of the file, the number of data). The dictonary will be written in a txt file together with the performances of the network
        
        Output (dictionary)
        """
        pass
