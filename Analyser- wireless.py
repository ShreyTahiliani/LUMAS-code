#LIVE SPECTRUM IMAGING ANALYSIS

#Libraries
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import io

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

#Wavelength Calibration
known_pixel_positions = [100, 500, 800, 1200]
known_wavelengths = [400, 500, 600, 700]
coefficients = np.polyfit(known_pixel_positions, known_wavelengths, 2)

def pixel_to_wavelength(pixel):
    return np.polyval(coefficients, pixel)

def find_elements_near_peak(peak_wavelength, tolerance=10):
    nearby_elements = []
    for element, wavelengths in element_data.items():
        for wl in wavelengths:
            if abs(wl - peak_wavelength) <= tolerance:
                nearby_elements.append((element, wl))
    return nearby_elements

#Stream Setup
STREAM_URL = "http://device-ip:8080/shot.jpg"

print("Starting live spectrum analysis...")

#Stabilization Variables
previous_intensity = None
stabilization_factor = 0.9  # Adjust to smooth more or less

#Real-time Processing
plt.ion()
fig, ax = plt.subplots()

while True:
    try:
        #Fetching the latest frame
        response = requests.get(STREAM_URL)
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)

        #Conversion to grayscale and extract the spectrum region
        gray_image = image.convert("L")
        spectrum_region = np.array(gray_image.crop((0, 100, gray_image.width, 200)))  # Adjust ROI

        #Calculation of intensity by averaging rows
        intensity = np.mean(spectrum_region, axis=0)

        #To stabilize intensity by averaging with previous frame
        if previous_intensity is not None:
            intensity = stabilization_factor * previous_intensity + (1 - stabilization_factor) * intensity
        previous_intensity = intensity

        #To smoothen the intensity values
        smoothed_intensity = savgol_filter(intensity, window_length=11, polyorder=2)

        #Calibrate pixel positions to wavelengths
        pixels = np.arange(len(smoothed_intensity))
        wavelengths = pixel_to_wavelength(pixels)

        #Detect peaks in the spectrum
        threshold = np.mean(smoothed_intensity) + np.std(smoothed_intensity) * 0.5
        peaks, _ = find_peaks(smoothed_intensity, height=threshold)
        peak_wavelengths = wavelengths[peaks]

        #Identify elements at each peak
        detected_elements = {}
        for peak_wl in peak_wavelengths:
            elements = find_elements_near_peak(peak_wl)
            if elements:
                detected_elements[peak_wl] = elements

        #Display Results
        ax.clear()
        ax.plot(wavelengths, smoothed_intensity, label="Spectrum", color="blue")
        ax.fill_between(wavelengths, smoothed_intensity, color="blue", alpha=0.1)  # Highlight area

        #Highlight RGB regions
        ax.axvspan(400, 500, color='blue', alpha=0.2, label="Blue Region")
        ax.axvspan(500, 600, color='green', alpha=0.2, label="Green Region")
        ax.axvspan(600, 700, color='red', alpha=0.2, label="Red Region")

        #Annotate detected peaks and elements
        for peak_wl, elements in detected_elements.items():
            element_text = ', '.join([f"{el} ({wl:.1f} nm)" for el, wl in elements])
            peak_index = np.where(wavelengths == peak_wl)[0]
            if peak_index.size > 0:
                peak_index = peak_index[0]
                ax.annotate(element_text, (peak_wl, smoothed_intensity[peak_index]),
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color="black",
                            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8))

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title("Live Spectrum Analysis")
        ax.legend()
        plt.pause(0.01)

    except KeyboardInterrupt:
        print("Stopping live analysis...")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue

plt.ioff()
plt.show()