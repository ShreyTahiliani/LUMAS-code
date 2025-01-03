#MANUAL SPECTRUM ANALYSIS

#Libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import find_peaks, savgol_filter

#To load the image into spectrum & convert it into RGB
try:
    image = Image.open('spectrum_image.jpg').convert('RGB')
    print("Image loaded successfully.")
except FileNotFoundError:
    print("Error: 'spectrum_image.jpg' not found. Make sure the file is in the correct directory.")
    exit()

#Defining an array
image_array = np.array(image)

time.sleep(0.9)

#Choice of row where spectrum would be clear
spectrum_row = image_array[100, :, :]  # Adjust this based on where the spectrum is most visible

#Conversion of RGB into Intensity to make an Intensity-Wavelength Graph
intensity = np.mean(spectrum_row, axis=1)
print("Intensity values calculated.")

#Smoothening Intensity values for noise reduction using Sav_Gol Filter 
smoothed_intensity = savgol_filter(intensity, window_length=11, polyorder=2)

#Calibrating pixel positions to wavelengths
known_pixel_positions = [100, 500, 800, 1200]  # Example pixel positions for known peaks
known_wavelengths = [400, 500, 600, 700]  # Corresponding wavelengths in nanometers
coefficients = np.polyfit(known_pixel_positions, known_wavelengths, 2)

#Function to convert pixel position to wavelength using the polynomial fit
def pixel_to_wavelength(pixel):
    return np.polyval(coefficients, pixel)

#To apply the calibration to all pixel positions to get the corresponding wavelengths
pixels = np.arange(len(smoothed_intensity))  # Create an array of pixel positions
wavelengths = pixel_to_wavelength(pixels)

#Threshold peak detection
threshold = np.mean(smoothed_intensity) + np.std(smoothed_intensity) * 0.5

#Upward Peak Detection
peaks, _ = find_peaks(smoothed_intensity, height=threshold)
peak_wavelengths = wavelengths[peaks]

#Elements with thier wavelengths
element_data = {
    "Hydrogen": [410.2, 434,486.1, 656.3],
    "Helium": [587.6, 468.6, 667.8],
    "Oxygen": [777.4, 844.6, 407],
    "Nitrogen": [399.5, 460.1],
    "Carbon": [247.9, 265.5, 357.7],
    "Sodium":  [589, 589.9],
    "Calcium": [393.4, 396.8, 422.2],
    "Magnesium": [518.4, 577],
    "Iron": [526.9, 532.8, 458.3],
    "Boron": [249.7, 257.9],
    "Aluminium": [396.1, 667.8],
    "Silicon": [288.1, 390.5, 410.3],
    "Sulphur": [921, 406.8],
    "Chromium": [425.4, 427.5],
    "Cobalt": [345.3, 350.5, 355.5],
    "Strontium": [460.7, 421.5, 407.8],
    "Radon": [508, 534.3],
    "Platinum": [360.3, 405.8, 304.3],
    "Silver": [328.1, 338.3, 481.3],
    "Ruthenium": [265.8, 373.1, 410.3],
    "Rhodium": [343.2, 373, 420.6],
    "Palladium": [341.4, 350.5, 379.8],
    "Tantalum": [260, 261.4, 277.1],
    "Niobium": [341.8, 347, 384.3],
    "Molybdenum": [314, 370, 385.5],
    "Rhenium": [335, 350.2, 406],
    "Osmium": [248.3, 278.6, 305.6],
    "Iridium": [238.3, 251.6, 291],
    "Tungsten": [312.3, 335, 400.9],
    "Uranium": [328.3, 367.3, 405],
    "Neodymium": [334.5, 354.9, 379.5],
    "Samarium": [343.1, 364.8, 401.9],
    "Europium": [420.3, 443, 552.1],
    "Gadolinium": [ 335, 363, 393],
    "Cerium": [404.7, 418.6, 422.7],
    "Lanthanum" : [327.7, 379.5, 407.4],
    "Neon": [585.2, 640.2],
    "Actinum": [339, 403],
    "Thorium": [401.9, 426.5, 433.6],
    "Plutonium": [239.3, 315.2],
    "Americium": [442, 548],
    "Curium": [250, 291],
    "Berkelium": [290, 315],
    "Californium": [404, 442],
    "Fermium": [283, 309],
    "Mendelevium": [271, 310],
    "Lawrencium": [340, 380],
    "Rutherfordium": [271, 289],
    "Dubnium": [278, 302],
    "Seaborgium": [267, 291],
    "Bohrium": [274, 295],
    "Hassium": [252, 270],
    "Lithium": [670.8, 610.3, 460.3],
    "Beryllium": [234.8, 313.1],
    "Fluorine": [685.6, 739.9],
    "Chlorine": [725.7, 858.6],
    "Argon": [696.5, 742.4],
    "Copper": [324.7, 510.6, 327.4],
    "Zinc": [213.9, 481],
    "Lead": [405.8, 440.6],
    "Nickel": [330.3, 341.5, 371],
    "Titanium": [334.2, 336.1, 376.1],
    "Manganese": [403.1, 404.4, 403.1],
    "Zirconium": [347.1, 339.6, 346.4],
    "Barium": [455.4, 493.4],
    "Radium": [407.8, 442],
    "Pottasium": [404.4, 769.9, 766.5],
    "Phosphorus": [253.4, 178.3]
}

#Setting Tolerance to show elements around the peak wavelength, such that closest elements are studied
def find_elements_near_peak(peak_wavelength, tolerance=10):
    nearby_elements = []
    for element, wavelengths in element_data.items():
        for wl in wavelengths:
            if abs(wl - peak_wavelength) <= tolerance:
                nearby_elements.append((element, wl))
    return nearby_elements

#Toggling Peaks along with data
clicked_peaks = {}

def on_peak_click(event, ax):
    if event.inaxes == ax:
        # Get the clicked wavelength from the x-axis
        clicked_wavelength = event.xdata
        # Find the nearest peak to the clicked point
        nearest_peak_idx = np.abs(peak_wavelengths - clicked_wavelength).argmin()
        nearest_peak_wavelength = peak_wavelengths[nearest_peak_idx]

        # Toggle visibility of element data
        if nearest_peak_wavelength in clicked_peaks:
            # Remove annotation if already clicked
            clicked_peaks[nearest_peak_wavelength].remove()
            del clicked_peaks[nearest_peak_wavelength]
        else:
            # Show element data if not clicked
            elements = find_elements_near_peak(nearest_peak_wavelength)
            if elements:
                text = '\n'.join([f'{el}: {wl} nm' for el, wl in elements])
                annotation = ax.annotate(text, (nearest_peak_wavelength, smoothed_intensity[peaks[nearest_peak_idx]]),
                                         textcoords="offset points", xytext=(0, 10), ha='center', color='blue')
                clicked_peaks[nearest_peak_wavelength] = annotation
            else:
                annotation = ax.annotate('No matching elements', (nearest_peak_wavelength, smoothed_intensity[peaks[nearest_peak_idx]]),
                                         textcoords="offset points", xytext=(0, 10), ha='center', color='red')
                clicked_peaks[nearest_peak_wavelength] = annotation
        plt.draw()

time.sleep(1.2)
print("Opening Color Map...")
time.sleep(0.1)

#Creation of Color map
def wavelength_to_rgb(wavelength):
    if wavelength < 380:
        wavelength = 380
    elif wavelength > 700:
        wavelength = 700

    if 380 <= wavelength <= 440:
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= wavelength <= 490:
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif 490 <= wavelength <= 510:
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= wavelength <= 645:
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif 645 <= wavelength <= 700:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0
    return (r, g, b)

#Generate the corresponding colors for the wavelengths in the visible spectrum
spectrum_colors = np.array([wavelength_to_rgb(wl) for wl in wavelengths])

#Plot the Spectrum with Colors
fig, ax = plt.subplots(figsize=(10, 5))

for i in range(len(wavelengths) - 1):
    ax.plot(wavelengths[i:i + 2], smoothed_intensity[i:i + 2], color=spectrum_colors[i], linewidth=2)

#Mark upward peaks
ax.plot(wavelengths[peaks], smoothed_intensity[peaks], "x", label="Upward Peaks", color='green')

#Connect click event to plot
fig.canvas.mpl_connect('button_press_event', lambda event: on_peak_click(event, ax))
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Smoothened Intensity')
ax.set_title('Spectrum with Element Detection')
ax.legend()

#Display()
plt.show()