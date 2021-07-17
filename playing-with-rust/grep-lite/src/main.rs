use clap::{App, Arg};
use regex::Regex;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

fn main() {
    let cmd_args = App::new("grep-lite")
        .version("1.0")
        .author("Daniel Kwatia Osei")
        .about(
            "Playing with rust after reading the \
    chapter 2 of Rust in Action",
        )
        .arg(
            Arg::with_name("Before")
                .short("B")
                .long("Before")
                .required(false)
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("After")
                .short("A")
                .long("After")
                .required(false)
                .takes_value(true)
                .default_value("0"),
        )
        .arg(
            Arg::with_name("pattern")
                .required(true)
                .help("Provides patterns to search for in text"),
        )
        .arg(
            Arg::with_name("file")
                .required(true)
                .help("Specifies the filename to search")
                .default_value(""),
        )
        .get_matches();

    //extract data from command line arguments
    let filename = cmd_args.value_of("file").unwrap();
    let pattern = cmd_args.value_of("pattern").unwrap();
    let re = Regex::new(pattern).unwrap();

    //
    let fh = File::open(filename).unwrap();
    let reader = BufReader::new(fh);

    //iterating through the lines
    for (line_no, line_) in reader.lines().enumerate() {
        let line = line_.unwrap();
        match re.find(&line) {
            Some(_) => {
                println!("{}: {}", line_no, line);
            }
            None => (),
        }
    }
}
