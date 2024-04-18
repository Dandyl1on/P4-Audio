import numpy as np
import matplotlib.pyplot as plt

def main():
    A = np.tile([-1, 1], 32768)
    A = np.concatenate((A, [1]))

    plt.plot(A)
    plt.show()

    B = np.fft.fft(A)
    B = B[1:]

    B = B[:len(B)//2]

    R = np.real(B)
    I = np.imag(B)

    RR = np.zeros((128, 256))
    II = np.zeros((128, 256))

    for k in range(128):
        RR[k,:] = R[256*k:256*k+256]
        II[k,:] = I[256*k:256*k+256]

    AA = np.concatenate((RR, II), axis=0)
    F = np.max(np.abs(AA))

    print(F)

if __name__ == "__main__":
    main()
