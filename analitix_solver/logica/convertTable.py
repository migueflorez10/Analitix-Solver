import pandas as pd
import os
from datetime import datetime
import shutil

def tableToText(titles, data, nameMethod):
    
    for i in range(1, len(titles)):
        titles[i] = f"          {titles[i]}"
        
    dfData = pd.DataFrame(data, columns=titles)
        
    nameFile = f'{nameMethod}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'
    route = getDirectory() + f'\\analitix_solver_tables\\'
    createDirectory(route)
    dfData.to_csv(f"{route}\\{nameFile}", index=False, sep='\t')
    
def getDirectory():
    System = os.name
    Download = None
    if System == 'posix':
        Download = os.path.join(os.path.expanduser("~"),"Downloads")
    elif System == 'nt':   
        Download = os.path.join(os.path.expanduser("~"),"Downloads")
    return Download

def createDirectory(route):
    if not os.path.exists(route):
        os.mkdir(route)
    else:
        pass
    