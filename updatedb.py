
from __future__ import print_function, division
import pyfits as pf
import logging

logging.basicConfig(level=logging.DEBUG)

# todo: force lower case


class AstroInstrumentInfo:
    """Object that has the information about a single instrument FITS field names.
    It does not access field values of a particular header.  It requires a first
    name for the instrument to be initialized, but then many alias or field values.

    Attribute dictionary 'fields' holds the different alternatives for a FITS header
    for a given instrument.  They are accessible through .get_header_alternatives()
    """

    def __repr__(self):
        return "<'{}'>".format(self.full_name)
    
    def get_fields(self):
        """Returns the fields available (those that have been initialized by
        .add_*() method)"""
        return [f for f in self.fields.keys() if f is not 'alias']
    
    def __getattr__(self, name):
        """Functions that fall into the format .first_second() are processed here,
        where first is the function name (currently count or add) and second is the
        field name in which to act"""
        
        try:
            fcn_prefix, field_name = name.split("_", 1)
        except ValueError:
            raise AttributeError("'{}' object has no attribute "
                                 "'{}'".format(self.__class__.__name__, name))

        def add_function(thing):
            if not isinstance(thing, (tuple, list)):
                thing = [thing]
            for t in thing:
                if field_name not in self.fields.keys():
                    logging.info("Initializing field '{}'".format(field_name))
                    self.fields[field_name] = []
                logging.info("adding alternative '{}' to '{}'".format(thing, field_name))
                list_of_things =  self.fields[field_name]
                if t not in list_of_things:
                    list_of_things.append(t)
            return self

        def count_function(obj):
            att = self.fields[field_name]
            if not isinstance(att, list):
                raise TypeError("Attribute to count is not a list. error in object "
                                "initialization")
            return len(att)

        valid_fcn_prefix={"count": count_function,
                          "add": add_function}
        
        if fcn_prefix not in valid_fcn_prefix:
            raise AttributeError("'{}' object has no attribute "
                                 "'{}'".format(self.__class__.__name__, name))

        return valid_fcn_prefix[fcn_prefix]

    
    def get_field_alternatives(self, field):
        """Returns all the alternatives for header value for a given instrument"""
        kws = self.fields[field]
        if not isinstance(kws, list):
            raise TypeError("request keyword {} is not a list in AstroInstrumentInfo.  It has not been initialized for '{}' instrument".format(field, self.full_name))
        return kws
    
        
    def __init__(self, name, full_name=None):
        """Mandatory to give a first name for the instrument, and an optional long name"""

        self.fields = {}
        self.add_alias(name)

        if full_name is None:
            full_name = name
        self.full_name = full_name

        self.id_dict = {}   # identifies by the correct combination of keyword and value
        self.id_field = []  # Identifies just by having this keyword appear

        self.ignore_dict = {}  # identifies frames to be ignored
        self.bias_dict = {}  # identifies bias with such keyword
        self.flat_dict = {}  # identifies flat with specified keyword

        def identity(x):
            return x
        self.caster = {'identity': identity}

    def is_bias(self, header):
        """Returns True if the header is identified as Bias"""
        if len(self.bias_dict) == 0:
            return False

        for kw, value in self.bias_dict.items():
            # logging.debug("Trying to identify BIAS with {}={}".format(kw, value))
            if kw in header and header[kw] == value:
                return True

        return False

    def is_flat(self, header):
        """Returns True if the header is identified as Flat"""
        if len(self.flat_dict) == 0:
            return False

        for kw, value in self.flat_dict.items():
            #logging.debug("Trying to identify Flat with {}={}".format(kw, value))
            if kw in header and header[kw] == value:
                return True

        return False

    def ignore(self, header):
        """Returns True if the header is identified as ignore"""
        if len(self.ignore_dict) == 0:
            return False

        for kw, value in self.ignore_dict.items():
            #logging.debug("Trying to identify ignore with {}={}".format(kw, value))
            if kw in header and header[kw] == value:
                return True

        return False

    def header_belongs(self, header):
        """Returns true if the header is identified as belonging to this instrument"""
        if self.count_id == 0:
            raise ValueError("AstroInstrument '{}' does not have any identification methods initialized)".format(self.name))

        for kw in self.id_field:
            #logging.debug("Trying to identify instrument '{}' with field '{}' ".format(self.name,kw))
            if kw in header:
                return True

        for kw,value in self.id_dict.items():
            #logging.debug("Trying to identify instrument '{}' with combination '{}={}' ".format(self.name, kw,value))
            if kw in header and header[kw].lower()==value:
                return True

        return False

    @property
    def name(self):
        """alias to full_name"""
        return self.full_name

    def add_ignore(self, **kwargs):
        for kw, value in kwargs.items():
            #logging.debug("IGNORE identified with {}={}".format(kw, value))
            self.ignore_dict[kw.lower()] = value

    def add_bias(self, **kwargs):
        for kw, value in kwargs.items():
            #logging.debug("BIAS identified with {}={}".format(kw, value))
            self.bias_dict[kw.lower()] = value

    def add_flat(self, **kwargs):
        for kw, value in kwargs.items():
            #logging.debug("FLAT identified with {}={}".format(kw, value))
            self.flat_dict[kw.lower()] = value

    def add_id(self, *args, **kwargs):
        """Add identification mechanism for this instrument. Either the existence of a particular field will tell, or a particular combination field=value"""
        for kw, value in kwargs.items():
            #logging.debug("Adding identification for instrument '{}': Combination '{}={}' ".format(self.name, kw,value))
            self.id_dict[kw.lower()] = value
        for kw in args:
            #logging.debug("Adding identification for instrument '{}': Field '{}' ".format(self.name, kw))
            if not isinstance(kw, str):
                raise TypeError("Only string is allowed as argument to add_id to search for a particular header ({})".format(kw))
            self.id_field.append(value)


    def set_cast(self, kw, function):
        """Adds recommended casting function to given keyword"""
        if kw in self.caster.keys():
            logging.warn("Overwriting cast function for keyword '{}' at instrument '{}'".format(kw, self.telescope))
        if not hasattr(function, '__call__'):
            raise TypeError("Second argument '{}' must be a callable function that accepts one parameter".format(function))
        self.caster[kw]=function

        
    def count_id(self):
        """Return the total number of identification methods"""
        return (len(self.id_dict)+len(self.id_field))
    

    def set_name(self, full_name):
        """Set the long name"""
        self.full_name =  full_name
        self.add_alias(full_name)

    def cast(self, name):
        """Cast recommended for particular field. Is up to the caller whether he wants to use it or not since this class does not access the header value"""
        if name not in self.caster:
            name='identity'
        return self.caster[name]

    


##########################################################
#
##################


class AstroHeaderInstrument(object):
    """Contains a header already identified as belonging to a certain instrument, its values can be accesed as
     attributes (and they are the only attributes visible) as long as they were initialized in the AstroInstrumetInfo
     with .add_*() """
    def __repr__(self):
        return "<FITS header for instrument '{}'>".format(self._instrument.full_name)

    def __dir__(self):
        return [self._instrument.get_fields(), 'instrument_name']


    def instrument_name(self):
        """Returns the name of the instrument"""
        return self._instrument.name

    def keywords(self, name, only_one=False):
        """Return the keyword alternatives"""
        kws = self._instrument.get_field_alternatives(name)
        if only_one:
            return kws[0]
        return kws

    def ignore(self):
        return self._instrument.ignore(self._header)

    def is_bias(self):
        return self._instrument.is_bias(self._header)

    def is_flat(self):
        return self._instrument.is_flat(self._header)

    def __init__(self, header, instrument):
        self._header = header
        self._instrument = instrument

        self.name = instrument.name
        self.full_name = instrument.full_name

    def __getattr__(self, name):
        """Returns casted field from FITS file"""
        fields = self._instrument.get_fields()
        if name not in fields:
            raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__.__name__, name))
        return self._instrument.cast(name)(self._find_kw(name))

    def __call__(self, header=None, hdu=0):
        if header is None:
            return self
        if isinstance(header, pf.header.Header):
            header = header
        elif isinstance(header, str):
            header = pf.open(header)[hdu].header
        elif isinstance(header, dp.AstroFile):
            header = header.readheader()
        else:
            raise TypeError("Header must by a pyfits header, a filename, or an astrofile")
        return AstroHeaderInstrument(header, self._instrument)
    
    def _find_kw(self,kw):
        """Finds the field's value without casting"""
        kws = self._instrument.get_field_alternatives(kw)
        
        for k in kws:
            if k in self._header:
                return self._header[k]
            
        logging.warn("Requested keyword {} not found for header in instrument {}.  Searched alternatives: {}".format(kw, self._instrument.name, kws))
        return None


###########################################
#
#############

        
class AstroInstrumentsInfo(object):
    """Class that brings together all the available instruments in database.  Basically, it purpose is to be fed the
     instrument characteristics, then be queried for header belonging to any.  Returns an AstroHeaderInstrument
     if successfull in identifying"""
    
    def __init__(self):
        #fill in defaults
        self.instruments = []

    def add_instrument(self, tel):
        if not isinstance(tel, AstroInstrumentInfo):
            raise TypeError("Only instruments of the class AstroInstrumentInfo can be added to AstroInstrumentsInfo")
        self.instruments.append(tel)

    def identify(self, header, hdu=0):

        if isinstance(header, pf.header.Header):
            header = header
        elif isinstance(header, str):
            header = pf.open(header)[hdu].header
        elif isinstance(header, dp.AstroFile):
            header = header.readheader()
        else:
            raise TypeError("Header must by a pyfits header, a filename, or an astrofile")
        
        found = None
        for inst in self.instruments:
            if inst.header_belongs(header):
                if found is not None:
                    raise ValueError("More than one instrument ({},{}) identifies with given"
                                     " header".format(found.instrument_name(), inst.name))
                found = AstroHeaderInstrument(header, inst)

        if found is None:
            logging.warn("Could not identify header as beloging to any of {}".format(self.instruments))
            return None

        return found

    # def identify(self, header, hdu=0):
    #
    #     instrument = identify_instrument(header, hdu)
    #     if instrument is None:
    #         return None
    #
    #     header_instrument = AstroHeaderInstrument(header, instrument)
    #     logging.debug("Found instrument '{}' for header".format(found.instrument_name()))
    #     return found

    def set_cast(self, kw, function):
        """Propagates casting for keyword to all instruments"""

        logging.warn("Propagating casting '{}' for '{}' to all instruments: {}".format(function, kw, self.instruments))

        for inst in self.instruments:
            inst.set_cast(kw, function)

######################################################
#
#################

import astropy.coordinates as apc

TInfo = AstroInstrumentsInfo()

danish = AstroInstrumentInfo('danish')
danish.add_alias('dk154')
danish.add_ra('orira')
danish.add_dec('oridec')
danish.add_object('object')
danish.add_jd('jd')
danish.add_exptime('exptime')
danish.add_id(telescop='dk-1.54')
danish.add_bias(imagetyp='BIAS')
danish.add_flat(imagetyp='FLAT')
danish.add_ignore(object='test')
danish.add_ignore(object='test_')
TInfo.add_instrument(danish)


import scipy as sp
import astropy.units as u
import astropy.time as apt
import astropy.coordinates as acoo
import dataproc as dp

class TransitObservation(object):


    def __init__(self, file_list, ref_frame=0, search_radius=5, max_delta_t=60):
        """Receives lots of files or an astrodir. Finds the coordinates of the first
         one, and then returns all those file with similar coordinate from within a range"""

        astrodir = dp.AstroDir(file_list)
        instrument = TInfo.identify(astrodir[ref_frame])

        ra0 = instrument().ra
        dec0 = instrument().dec
        target0 = instrument().object
        jdfield = instrument().keywords('jd', True)
        coo0 = acoo.SkyCoord(ra0, dec0, frame='icrs', unit=(u.hourangle, u.deg))

        in_file = []
        out_file = []
        times = []
        biases = []
        flats = []

        images = astrodir.sort(jdfield)
        for i in range(len(images)):
            image = images[i]
            inst_img = instrument(image)
#            logging.debug("Finding image {}".format(image))

            if inst_img.is_bias():
                biases.append(image)
                continue
            if inst_img.is_flat():
                flats.append(image)
                continue
            if inst_img.ignore():
                continue

            ra = inst_img.ra
            dec = inst_img.dec
            coo = acoo.SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))
            target = inst_img.object
            time = inst_img.jd

            sep = coo0.separation(coo).arcminute

            if len(in_file):
                if (sep > search_radius) or\
                   (time - times[-1] > max_delta_t):
                    break

            if sep < search_radius:
                in_file.append(image)
                if target != target0:
                    logging.warning("There are more than two object names: '{}' & '{}'".format(target0, target))
                times.append(time)
            else:
                out_file.append(image)

        if i+1 < len(images):
            danish.add_ignore(object='test')
            out_file.extend([img for img in images[i+1:]])

        delta_t = sp.array(times)[1:]-sp.array(times)[:-1]

        self.delta = {"max": max(delta_t) if len(delta_t) else 0,
                      "median": sp.median(delta_t),
                      "range": times[-1] - times[0],
                      "center": (times[-1] + times[0])/2,
                      }
        self.files = in_file
        self.bias = biases
        self.flat = flats
        self._no_files = out_file
        self.target = target0

    def __repr__(self):
        return "Transit by '{}' ({} frames)".format(self.target,
                                                    len(self.files),
                                                    )

    def info(self):
        return "Transit by '{}' ({} frames)\n" \
               " average between frames: {:.1f}m\n" \
               " observing block: {:.1f}h\n center: {}\n" \
               " calibration: {} bias, {} flats".format(self.target,
                                                        len(self.files),
                                                        self.delta["median"]*24*60,
                                                        self.delta["range"]*24,
                                                        apt.Time(self.delta["center"],
                                                                 format='jd').isot,
                                                        len(self.bias),
                                                        len(self.flat),
                                                        )

    def remainder(self):
        return dp.AstroDir(self._no_files)

    def __len__(self):
        return len(self.files)
        

import os
import sys

def find_transit(basedir):
    dirs = [x[0] for x in os.walk(basedir) if 'focus' not in x[0]]
    store = []

    for d in dirs:
        print("examining {}: ".format(d), end="")
        files = dp.AstroDir(d+"/*fits*")
        while(len(files)):
            print(".", end='')
            sys.stdout.flush()
            tr = TransitObservation(files)
            if len(tr):
                store.append(tr)
            files = tr.remainder()
        print("done")

    return store







