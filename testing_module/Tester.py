from abc import ABC, abstractmethod
from tqdm import tqdm
from skimage.color import rgb2gray
import cupy


class Tester(ABC):
   @abstractmethod
   def init(self, images,time_sync_function,*args, **kwargs):
      raise NotImplementedError()
   @abstractmethod
   def calculate_time(self,device_func, *args, **kwargs):
      raise NotImplementedError()
     
class FunctionTester(Tester):
   def init(self, images:list, time_sync_function:callable,function:callable):
      self.function = function
      self.images = images
      self.time_sync_function = time_sync_function
   def calculate_time(self,device_func:callable, operation_name = None):
      print(operation_name)
      if operation_name == "Sliding Window Detection":
         for image in self.images:
            if len(image.shape) == 3:
               print(type(image))
               image = rgb2gray(image)

      t1 = self.time_sync_function()
      for image in tqdm(self.images):
         image = device_func(image)
         self.function(image)
      t2 = self.time_sync_function()
      return t2-t1
   

      
      