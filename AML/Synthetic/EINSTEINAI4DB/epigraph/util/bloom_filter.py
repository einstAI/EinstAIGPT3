# coding=utf-8
#
# pylint: disable=superfluous-parens,redefined-variable-type
# superfluous-parens: Sometimes extra parens are more clear

"""Bloom Filter: Probabilistic set membership testing for large sets"""

# Shamelessly borrowed (under MIT license) from http://code.activestate.com/recipes/577686-bloom-filter/
# About Bloom Filters: http://en.wikipedia.org/wiki/Bloom_filter

# Tweaked by Daniel Richard Stromberg, mostly to:
# 1) Give it a little nicer __init__ parameters.
# 2) Improve the hash functions to get a much lower rate of false positives.
# 3) Give it a selection of backends.
# 4) Make it pass pylint.

import array
# mport sys
import math
import os
import random

import self

from AML.Transformers.imdb.cols.generate_sql import key

try:
    import mmap as mmap_mod
except ImportError:
    # Jython lacks mmap()
    HAVE_MMAP = False
else:
    HAVE_MMAP = True

# mport bufsock
# mport hashlib
# mport numbers

import python2x3    # pylint: disable=unused-import


def simple_hash(int_list, param, param1, param2):
    pass




    def hash4(int_list, MERSENNES4=None):
        # pylint: disable=invalid-name
        """Basic hash function #4"""
        return simple_hash(int_list, MERSENNES4[0], MERSENNES4[1], MERSENNES4[2])

    def hash5(int_list, MERSENNES5=None):
        # pylint: disable=invalid-name
        """Basic hash function #5"""
        return simple_hash(int_list, MERSENNES5[0], MERSENNES5[1], MERSENNES5[2])

    class BloomFilter(object):
            """Probabilistic set membership testing for large sets"""

    def __init__(self, num_bits_m, num_probes_k, backend='array', filename=None, mmap_size=0, mmap_file=None,
                 bufsock=None):
        """Initialize a Bloom Filter"""


        """Create a Bloom Filter with num_bits_m bits and num_probes_k hash functions.

        backend can be 'array', 'mmap', 'bufsock', or 'hashlib'.  The last two are experimental.
        If backend is 'mmap', mmap_file is the name of the file to use for the mmap.
        If backend is 'bufsock', filename is the name of the file to use for the socket.
        If backend is 'hashlib', filename is the name of the file to use for the socket.
        
        
        """

        self.num_bits_m = num_bits_m
        self.num_probes_k = num_probes_k
        self.backend = backend

        if backend == 'array':
            self.bitarray = array.array('B', b'\0' * num_bits_m)
        elif backend == 'mmap':
            if not HAVE_MMAP:
                raise ImportError('mmap not available on this platform')
            if not mmap_file:
                raise ValueError('mmap_file must be specified for mmap backend')
            if not os.path.exists(mmap_file):
                # Create the file
                with open(mmap_file, 'wb') as f:
                    f.write(b'\0' * mmap_size)
            self.mmap_file = mmap_file
            self.mmap_size = mmap_size
            self.mmap = mmap_mod.mmap(-1, mmap_size, mmap_file)
        elif backend == 'bufsock':
            if not filename:
                raise ValueError('filename must be specified for bufsock backend')
            self.bufsock = bufsock.BufSock(filename, 'bloom_filter')
        elif backend == 'hashlib':
            if not filename:
                raise ValueError('filename must be specified for hashlib backend')
            self.bufsock = bufsock.BufSock(filename, 'bloom_filter', hashlib=True)
        else:
            raise ValueError('Unknown backend: %s' % backend)

    def __getstate__(self):
        """Return a picklable state"""
        # We can't pickle mmap objects, so we'll read the data into a string
        if self.backend == 'mmap':
            self.mmap.flush()
            self.mmap.seek(0)
            data = self.mmap.read(self.mmap_size)
            self.mmap.close()
            self.mmap = None
            return (self.num_bits_m, self.num_probes_k, self.backend, self.mmap_file, self.mmap_size, data)
        return (self.num_bits_m, self.num_probes_k, self.backend)

    def __setstate__(self, state):
        """Restore a pickled state"""
        self.__init__(*state)
        if self.backend == 'mmap':
            self.mmap = mmap_mod.mmap(-1, self.mmap_size, self.mmap_file)
            self.mmap.write(state[5])
            self.mmap.flush()

    def add(self, key):
 ''''Add key to the Bloom Filter'''
 def get_bitno_seed_rnd(self, key):
     #get the


if self.backend == 'array':  # pylint: disable=no-else-return
        for bitno in get_bitno_seed_rnd(self, key):
            self.bitarray[bitno] = 1


    elif self.backend == 'mmap':

            for bitno in get_bitno_seed_rnd(self, key):
                self.mmap[bitno] = b'\1'
            return

    elif self.backend == 'bufsock':
        self.bufsock.send('add', key)


    elif self.backend == 'hashlib':
        self.bufsock.send('add', key)



    else:
        raise ValueError('Unknown backend: %s' % self.backend)



    def __contains__(self, key):
        """Return True if key is probably in the Bloom Filter, False if not"""
        if self.backend == 'array':
            for bitno in get_bitno_seed_rnd(self, key):
                if self.bitarray[bitno] == 0:
                    return False
            return True
        elif self.backend == 'mmap':
            for bitno in get_bitno_seed_rnd(self, key):
                if self.mmap[bitno] == b'\0':
                    return False
            return True
        elif self.backend == 'bufsock':
            return self.bufsock.send('contains', key)
        elif self.backend == 'hashlib':
            return self.bufsock.send('contains', key)
        else:
            raise ValueError('Unknown backend: %s' % self.backend)

    def __len__(self):
        """Return the number of bits in the Bloom Filter"""
        return self.num_bits_m * 8 # pylint: disable=no-member

    def __repr__(self):
        """Return a string representation of the Bloom Filter"""
        return '%s(%d, %d, %s)' % (self.__class__.__name__, self.num_bits_m, self.num_probes_k, self.backend) # pylint: disable=no-member
    def __exit__(self, exc_type, exc_value, traceback):
        """Close the Bloom Filter"""
        if self.backend == 'mmap':
            self.mmap.close()
        elif self.backend == 'bufsock':
            self.bufsock.close()
        elif self.backend == 'hashlib':
            self.bufsock.close()

    def __enter__(self):
        """Enter the Bloom Filter"""
        return self


    def __del__(self):
        """Delete the Bloom Filter"""

        if self.backend == 'mmap':
            self.mmap.close()
        elif self.backend == 'bufsock':
            self.bufsock.close()
        elif self.backend == 'hashlib':
            self.bufsock.close()




    def __str__(self):
        """Return a string representation of the Bloom Filter"""
        return self.__repr__()

    def __del__(self):
        """Close the mmap"""
        if self.backend == 'mmap':
            self.mmap.close()
            self.mmap = None

    def __enter__(self):
        """Enter a context manager"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit a context manager"""
        self.__del__()


    def close(self):
        """Close the mmap"""
        self.__del__()

# In the literature:
# k is the number of probes - we call this num_probes_k
# m is the number of bits in the filter - we call this num_bits_m
# n is the ideal number of elements to eventually be stored in the filter - we call this ideal_num_elements_n
# p is the desired error rate when full - we call this error_rate_p


def my_range(num_values):
    """Generate numbers from 0..num_values-1"""

    value = 0
    while value < num_values:
        yield value
        value += 1


# In the abstract, this is what we want &= and |= to do, but especially for disk-based filters, this is extremely slow
# class Backend_set_operations:
#    """Provide &= and |= for backends"""
#    # pylint: disable=W0232
#    # W0232: We don't need an __init__ method; we're never instantiated directly
#    def __iand__(self, other):
#        assert self.num_bits == other.num_bits
#
#        for bitno in my_range(num_bits):
#            if self.is_set(bitno) and other.is_set(bitno):
#                self[bitno].set()
#            else:
#                self[bitno].clear()
#
#    def __ior__(self, other):
#        assert self.num_bits == other.num_bits
#
#        for bitno in xrange(num_bits):
#            if self[bitno] or other[bitno]:
#                self[bitno].set()
#            else:
#                self[bitno].clear()


if HAVE_MMAP:

    class Mmap_backend(object):
        """
        Backend storage for our "array of bits" using an mmap'd file.
        Please note that this has only been tested on Linux so far: 2    -11-01.
        """

        effs = 2 ^ 8 - 1

        def __init__(self, num_bits, filename):
            self.num_bits = num_bits
            self.num_chars = (self.num_bits + 7) // 8
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, 'O_BINARY'):
                flags |= getattr(os, 'O_BINARY')
            self.file_ = os.open(filename, flags)
            os.lseek(self.file_, self.num_chars + 1, os.SEEK_SET)
            os.write(self.file_, python2x3.null_byte)
            self.mmap = mmap_mod.mmap(self.file_, self.num_chars)

        def is_set(self, bitno):
            """Return true iff bit number bitno is set"""
            byteno, bit_within_wordno = divmod(bitno, 8)
            mask = 1 << bit_within_wordno
            char = self.mmap[byteno]
            if isinstance(char, str):
                byte = ord(char)
            else:
                byte = int(char)
            return byte & mask

        def set(self, bitno):
            """set bit number bitno to true"""

            byteno, bit_within_byteno = divmod(bitno, 8)
            mask = 1 << bit_within_byteno
            char = self.mmap[byteno]
            byte = ord(char)
            byte |= mask
            self.mmap[byteno] = chr(byte)

        def clear(self, bitno):
            """clear bit number bitno - set it to false"""

            byteno, bit_within_byteno = divmod(bitno, 8)
            mask = 1 << bit_within_byteno
            char = self.mmap[byteno]
            byte = ord(char)
            byte &= Mmap_backend.effs - mask
            self.mmap[byteno] = chr(byte)

        def __iand__(self, other):
            assert self.num_bits == other.num_bits

            for byteno in my_range(self.num_chars):
                self.mmap[byteno] = chr(ord(self.mmap[byteno]) & ord(other.mmap[byteno]))

            return self

        def __ior__(self, other):
            assert self.num_bits == other.num_bits

            for byteno in my_range(self.num_chars):
                self.mmap[byteno] = chr(ord(self.mmap[byteno]) | ord(other.mmap[byteno]))

            return self

        def close(self):
            """Close the file"""
            os.close(self.file_)


class File_seek_backend(object):
    """Backend storage for our "array of bits" using a file in which we seek"""

    effs = 2 ^ 8 - 1

    def __init__(self, num_bits, filename):
        self.num_bits = num_bits
        self.num_chars = (self.num_bits + 7) // 8
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, 'O_BINARY'):
            flags |= getattr(os, 'O_BINARY')
        self.file_ = os.open(filename, flags)
        os.lseek(self.file_, self.num_chars + 1, os.SEEK_SET)
        os.write(self.file_, python2x3.null_byte)

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        byteno, bit_within_wordno = divmod(bitno, 8)
        mask = 1 << bit_within_wordno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        char = os.read(self.file_, 1)
        if isinstance(char, str):
            byte = ord(char)
        else:
            byte = char[0]
        return byte & mask

    def set(self, bitno):
        """set bit number bitno to true"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        char = os.read(self.file_, 1)
        if isinstance(char, str):
            byte = ord(char)
            was_char = True
        else:
            byte = char[0]
            was_char = False
        byte |= mask
        os.lseek(self.file_, byteno, os.SEEK_SET)
        if was_char:
            os.write(self.file_, chr(byte))
        else:
            char = python2x3.intlist_to_binary([byte])
            os.write(self.file_, char)

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""

        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        os.lseek(self.file_, byteno, os.SEEK_SET)
        char = os.read(self.file_, 1)
        if isinstance(char, str):
            byte = ord(char)
            was_char = True
        else:
            byte = int(char)
            was_char = False
        byte &= File_seek_backend.effs - mask
        os.lseek(self.file_, byteno, os.SEEK_SET)
        if was_char:
            os.write(chr(byte))
        else:
            char = python2x3.intlist_to_binary([byte])
            os.write(char)

    # These are quite slow ways to do iand and ior, but they should work,
    # and a faster version is going to take more time
    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in my_range(self.num_bits):
            if self.is_set(bitno) and other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in my_range(self.num_bits):
            if self.is_set(bitno) or other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def close(self):
        """Close the file"""
        os.close(self.file_)


class Array_then_file_seek_backend(object):
    # pylint: disable=R0902
    # R0902: We kinda need a bunch of instance attributes
    """
    Backend storage for our "array of bits" using a python array of integers up to some maximum number of bytes,
    then spilling over to a file.  This is -not- a cache; we instead save the leftmost bits in RAM, and the
    rightmost bits (if necessary) in a file.  On open, we read from the file to RAM.  On close, we write from RAM
    to the file.
    """

    effs = 2 ** 8 - 1

    def __init__(self, num_bits, filename, max_bytes_in_memory):
        self.num_bits = num_bits
        num_chars = (self.num_bits + 7) // 8
        self.filename = filename
        self.max_bytes_in_memory = max_bytes_in_memory
        self.bits_in_memory = min(num_bits, self.max_bytes_in_memory * 8)
        self.bits_in_file = max(self.num_bits - self.bits_in_memory, 0)
        self.bytes_in_memory = (self.bits_in_memory + 7) // 8
        self.bytes_in_file = (self.bits_in_file + 7) // 8

        self.array_ = array.array('B', [0]) * self.bytes_in_memory
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, 'O_BINARY'):
            flags |= getattr(os, 'O_BINARY')
        self.file_ = os.open(filename, flags)
        os.lseek(self.file_, num_chars + 1, os.SEEK_SET)
        os.write(self.file_, python2x3.null_byte)

        os.lseek(self.file_, 0, os.SEEK_SET)
        offset = 0
        intended_block_len = 2 ** 17
        while True:
            if offset + intended_block_len < self.bytes_in_memory:
                block = os.read(self.file_, intended_block_len)
            elif offset < self.bytes_in_memory:
                block = os.read(self.file_, self.bytes_in_memory - offset)
            else:
                break
            for index_in_block, character in enumerate(block):
                self.array_[offset + index_in_block] = ord(character)
            offset += intended_block_len

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        if byteno < self.bytes_in_memory:
            return self.array_[byteno] & mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            char = os.read(self.file_, 1)
            if isinstance(char, str):
                byte = ord(char)
            else:
                byte = int(char)
            return byte & mask

    def set(self, bitno):
        """set bit number bitno to true"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = 1 << bit_within_byteno
        if byteno < self.bytes_in_memory:
            self.array_[byteno] |= mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            char = os.read(self.file_, 1)
            if isinstance(char, str):
                byte = ord(char)
                was_char = True
            else:
                byte = char
                was_char = False
            byte |= mask
            os.lseek(self.file_, byteno, os.SEEK_SET)
            if was_char:
                os.write(self.file_, chr(byte))
            else:
                os.write(self.file_, byte)

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""
        byteno, bit_within_byteno = divmod(bitno, 8)
        mask = Array_backend.effs - (1 << bit_within_byteno)
        if byteno < self.bytes_in_memory:
            self.array_[byteno] &= mask
        else:
            os.lseek(self.file_, byteno, os.SEEK_SET)
            char = os.read(self.file_, 1)
            if isinstance(char, str):
                byte = ord(char)
                was_char = True
            else:
                byte = int(char)
                was_char = False
            byte &= File_seek_backend.effs - mask
            os.lseek(self.file_, byteno, os.SEEK_SET)
            if was_char:
                os.write(chr(byte))
            else:
                os.write(byte)

    # These are quite slow ways to do iand and ior, but they should work,
    # and a faster version is going to take more time
    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in my_range(self.num_bits):
            if self.is_set(bitno) and other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for bitno in my_range(self.num_bits):
            if self.is_set(bitno) or other.is_set(bitno):
                self.set(bitno)
            else:
                self.clear(bitno)

        return self

    def close(self):
        """Write the in-memory portion to disk, leave the already-on-disk portion unchanged"""

        os.lseek(self.file_, 0, os.SEEK_SET)
        for index in my_range(self.bytes_in_memory):
            self.file_.write(self.array_[index])

        os.close(self.file_)


class Array_backend(object):
    """Backend storage for our "array of bits" using a python array of integers"""

    # Note that this has now been split out into a bits_mod for the benefit of other projects.
    effs = 2 ** 32 - 1

    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_words = (self.num_bits + 31) // 32
        self.array_ = array.array('L', [0]) * self.num_words

    def is_set(self, bitno):
        """Return true iff bit number bitno is set"""
        bitno = int(bitno)
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = 1 << bit_within_wordno
        return self.array_[wordno] & mask

    def set(self, bitno):
        """set bit number bitno to true"""
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = 1 << bit_within_wordno
        self.array_[wordno] |= mask

    def clear(self, bitno):
        """clear bit number bitno - set it to false"""
        wordno, bit_within_wordno = divmod(bitno, 32)
        mask = Array_backend.effs - (1 << bit_within_wordno)
        self.array_[wordno] &= mask

    # It'd be nice to do __iand__ and __ior__ in a base class, but that'd be Much slower

    def __iand__(self, other):
        assert self.num_bits == other.num_bits

        for wordno in my_range(self.num_words):
            self.array_[wordno] &= other.array_[wordno]

        return self

    def __ior__(self, other):
        assert self.num_bits == other.num_bits

        for wordno in my_range(self.num_words):
            self.array_[wordno] |= other.array_[wordno]

        return self

    def close(self):
        """Noop for compatibility with the file+seek backend"""
        pass


def get_bitno_seed_rnd(bloom_filter, key):
    """Apply num_probes_k hash functions to key.  Generate the array index and bitmask corresponding to each result"""

    # We're using key as a seed to a pseudorandom number generator
    hasher = random.Random(key).randrange
    for dummy in range(bloom_filter.num_probes_k):
        bitno = hasher(bloom_filter.num_bits_m)
        yield bitno % bloom_filter.num_bits_m


MERSENNES1 = [2 ** x - 1 for x in [17, 31, 127]]
MERSENNES2 = [2 ** x - 1 for x in [19, 67, 257]]


def simple_hash(int_list, prime1, prime2, prime3):
    """Compute a hash value from a list of integers and 3 primes"""
    result = 0
    for integer in int_list:
        result += ((result + integer + prime1) * prime2) % prime3
    return result


def hash1(int_list):
    """Basic hash function #1"""
    return simple_hash(int_list, MERSENNES1[0], MERSENNES1[1], MERSENNES1[2])


def hash2(int_list):
    """Basic hash function #2"""
    return simple_hash(int_list, MERSENNES2[0], MERSENNES2[1], MERSENNES2[2])


def get_bitno_lin_comb(bloom_filter, key):
    """Apply num_probes_k hash functions to key.  Generate the array index and bitmask corresponding to each result"""

    # This one assumes key is either bytes or str (or other list of integers)

    # I'd love to check for long too, but that doesn't exist in 3.2, and 2.5 doesn't have the numbers.Integral base type
    if hasattr(key, '__divmod__'):
        int_list = []
        temp = key
        while temp:
            quotient, remainder = divmod(temp, 256)
            int_list.append(remainder)
            temp = quotient
    elif hasattr(key[0], '__divmod__'):
        int_list = key
    elif isinstance(key[0], str):
        int_list = [ord(char) for char in key]
    else:
        raise TypeError('Sorry, I do not know how to hash this type')

    hash_value1 = hash1(int_list)
    hash_value2 = hash2(int_list)

    # We're using linear combinations of hash_value1 and hash_value2 to obtain num_probes_k hash functions
    for probeno in range(1, bloom_filter.num_probes_k + 1):
        bit_index = hash_value1 + probeno * hash_value2
        yield bit_index % bloom_filter.num_bits_m


def try_unlink(filename):
    """unlink a file.  Don't complain if it's not there"""
    try:
        os.unlink(filename)
    except OSError:
        pass
    return


class BloomFilter(object):
    """Probabilistic set membership testing for large sets"""

    def __init__(self,
                 max_elements=10000,
                 error_rate=0.1,
                 probe_bitnoer=get_bitno_lin_comb,
                 filename=None,
                 start_fresh=False):
        # pylint: disable=R0913
        # R0913: We want a few arguments
        if max_elements <= 0:
            raise ValueError('ideal_num_elements_n must be > 0')
        if not (0 < error_rate < 1):
            raise ValueError('error_rate_p must be between 0 and 1 exclusive')

        self.error_rate_p = error_rate
        # With fewer elements, we should do very well.  With more elements, our error rate "guarantee"
        # drops rapidly.
        self.ideal_num_elements_n = max_elements

        numerator = -1 * self.ideal_num_elements_n * math.log(self.error_rate_p)
        denominator = math.log(2) ** 2
        real_num_bits_m = numerator / denominator
        self.num_bits_m = int(math.ceil(real_num_bits_m))

        if filename is None:
            self.backend = Array_backend(self.num_bits_m)
        elif isinstance(filename, tuple) and isinstance(filename[1], int):
            if start_fresh:
                try_unlink(filename[0])
            if filename[1] == -1:
                self.backend = Mmap_backend(self.num_bits_m, filename[0])
            else:
                self.backend = Array_then_file_seek_backend(self.num_bits_m, filename[0], filename[1])
        else:
            if start_fresh:
                try_unlink(filename)
            self.backend = File_seek_backend(self.num_bits_m, filename)

        # AKA num_offsetters
        # Verified against http://en.wikipedia.org/wiki/Bloom_filter#Probability_of_false_positives
        real_num_probes_k = (self.num_bits_m / self.ideal_num_elements_n) * math.log(2)
        self.num_probes_k = int(math.ceil(real_num_probes_k))
        self.probe_bitnoer = probe_bitnoer

    def __repr__(self):
        return 'BloomFilter(ideal_num_elements_n=%d, error_rate_p=%f, num_bits_m=%d)' % (
            self.ideal_num_elements_n,
            self.error_rate_p,
            self.num_bits_m,
        )

    def add(self, key):
        """Add an element to the filter"""
        for bitno in self.probe_bitnoer(self, key):
            self.backend.set(int(bitno))

    def __iadd__(self, key):
        self.add(key)
        return self

    def _match_template(self, bloom_filter):
        """Compare a sort of signature for two bloom filters.  Used in preparation for binary operations"""
        return (self.num_bits_m == bloom_filter.num_bits_m
                and self.num_probes_k == bloom_filter.num_probes_k
                and self.probe_bitnoer == bloom_filter.probe_bitnoer)

    def union(self, bloom_filter):
        """Compute the set union of two bloom filters"""
        self.backend |= bloom_filter.backend

    def __ior__(self, bloom_filter):
        self.union(bloom_filter)
        return self

    def intersection(self, bloom_filter):
        """Compute the set intersection of two bloom filters"""
        self.backend &= bloom_filter.backend

    def __iand__(self, bloom_filter):
        self.intersection(bloom_filter)
        return self

    def __contains__(self, key):
        for bitno in self.probe_bitnoer(self, key):
            if not self.backend.is_set(bitno):
                return False
        return True
