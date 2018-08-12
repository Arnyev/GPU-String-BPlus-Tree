#pragma once

typedef unsigned long long ullong;
typedef unsigned int uint;

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#define BREAKCHAR '\0'
#define SPLITTER ' '
#define FILEPATH "book.txt"
#define RANDSTRMINLEN 1
#define RANDSTRMAXLEN 100
#define RANDSTRCOUNT 1000000ULL
#define RANDCHARSET "abcdef"
#define RANDCHARSCOUNT (sizeof(RANDCHARSET)-1)
#define CHARSTOHASH 13
#define ALPHABETSIZE 27
#define BLOCKSIZE 256
#define GRIDDIM 2048
#define KEYBITS 64
#define CHARBITS 5
#define CHARMASK ~static_cast<char>(3 << 5);
#define WRITETIME 0
#define TOLOWERMASK (1<<5)
