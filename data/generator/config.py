"""

Returns:[]
"""
class Config:
    file_path = '/home/postscript/Workspace/DS/Workspace/Project/Code/stacksumm/data/interim/'
    count = 1000

    def getFullPath(self,filename):
        return self.file_path + filename
