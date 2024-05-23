from abc import ABC, abstractmethod


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
   def calculate_time(self,device_func:callable):
      t1 = self.time_sync_function()
      for image in self.images:
         image = device_func(image)
         self.function(image)
      t2 = self.time_sync_function()
      return t2-t1
   

      
      