from django.db import models

# Create your models here.

from django.db import models
from django.contrib.auth.models import User

class Transactions(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    category = models.CharField(max_length=50)

    def _str_(self):
        return f"{self.name} - {self.amount} - {self.category}"