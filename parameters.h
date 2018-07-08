#pragma once

typedef unsigned char uchar;
typedef unsigned long long ullong;
typedef unsigned int uint;

#define BREAKCHAR '\0'
#define SPLITTER ' '
#define FILEPATH "book.txt"
#define RANDSTRMAXLEN 128
#define RANDSTRCOUNT 10000000
#define RANDCHARSET "ab"
#define RANDCHARSCOUNT (sizeof(RANDCHARSET)-1)
#define CHARSTOHASH 13
#define ALPHABETSIZE 27
#define ASCIILOWSTART 96
#define ASCIIUPSTART 64
#define BLOCKSIZE 512
#define KEYBITS 64
#define CHARBITS 5
#define CHARMASK ~static_cast<uchar>(3 << 5);
#define WRITETIME 1
