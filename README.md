# Solar_limb_darkening

#Yr3 Astro lab Solar limb darkening experiment 
Analysing solar disc pixel count data 

Adjust the scan range and step size for each coefficient, depending on how precise you want it to be. 

This code automatically crops the disc region and computes the darkening coefficients.
Make sure there is sufficient data for the background (i.e. regions beyond the disc) for the auto-crop to function properly. 
You may adjust the std cut limit to select the amount of data wanted. Look at the cropped curve shown in Fig 1 to do the adjustment. 
If it doesn't work, switch off the auto-crop and manually enter the limits. 

This code generates estimations for the coefficients by, first, fitting a linear curve to it, before doing a fine search for the coefficients. 

The final fitted curve with the output coefficients is shown in Fig 5. Check if it makes sense. 

P.S. If anyone can make a smarter search algorithm instead of doing just a "blind scan" that would be much appreciated. 
