/**
 * @file command_parse.cuh
 * @author Jasper Jeuken
 * @brief Defines a command line argument parser for the raytracer
 */
#ifndef COMMAND_PARSE_H
#define COMMAND_PARSE_H

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <set>
#include "utility.cuh"


/**
 * @struct parsed_args
 * @brief Parsed command line arguments
 */
struct parsed_args {
    std::string scene_file; ///< Path to the scene file
    bool preview; ///< Whether to show a preview window
    std::string output_path; ///< Path to the output directory
    std::string output_format; ///< Output image format
    std::set<std::string> render_passes; ///< Selected render passes
};

/**
 * @brief Splits a string by a delimiter
 * 
 * @param[in] s String to split
 * @param[in] delimiter Delimiter to split by
 * @return Split elements
 */
std::vector<std::string> split(const std::string& s, char delimiter = ' ') {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief Parses a string of render passes
 * 
 * @param[in] value String of render passes
 * @return Selected render passes
 */
std::set<std::string> parse_passes(const std::string& value) {
    std::set<std::string> selected;
    std::set<std::string> valid = {"color", "albedo", "emission", "normal", "depth", "opacity", "denoised"};

    if (value.empty()) {
        return valid;
    }

    std::vector<std::string> tokens = split(value, ',');
    for (const std::string& token : tokens) {
        // Include all passes
        if (token == "all") {
            for (const std::string& pass : valid) {
                selected.insert(pass);
            }

        // Exclude specific pass
        } else if (!token.empty() && token[0] == '!') {
            std::string negated = token.substr(1);
            if (valid.count(negated)) {
                selected.erase(negated);
            } else {
                std::cerr << "Invalid render pass: " << negated << std::endl;
                exit(1);
            }

        // Include specific pass
        } else if (valid.count(token)) {
            selected.insert(token);
        } else {
            std::cerr << "Invalid render pass: " << token << std::endl;
            exit(1);
        }
    }
    return selected;
}


/**
 * @class command_parser
 * @brief Command line argument parser
 * @see https://stackoverflow.com/a/868894
 */
class command_parser {
public:

    /**
     * @brief Construct a new command_parser object
     * 
     * @param[in] argc Number of command line arguments
     * @param[in] argv Command line arguments
     */
    command_parser (int &argc, char **argv) {
        for (int i=1; i < argc; ++i) {
            this->tokens.push_back(std::string(argv[i]));
        }
    }

    /**
     * @brief Get the value of a command line argument
     * 
     * @param[in] option Command line argument
     * @return Value of the command line argument
     */
    const std::string& get_cmd_option(const std::string &option) const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    /**
     * @brief Check if a command line argument exists
     * 
     * @param[in] option Command line argument
     * @return True if the command line argument exists
     */
    bool cmd_option_exists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option)
               != this->tokens.end();
    }

private:
    std::vector<std::string> tokens; ///< Command line tokens
};


/**
 * @brief Parses command line arguments
 * 
 * @param[in] argc Number of command line arguments
 * @param[in] argv Command line arguments
 * @return Parsed command line arguments
 */
parsed_args parse_command_line_args(int argc, char** argv) {
    command_parser parser(argc, argv);
    parsed_args args;

    // Cmd arg: scene file
    args.scene_file = parser.get_cmd_option("-s");
    if (args.scene_file.empty()) {
        error_with_message("No scene file specified");
    }

    // Cmd arg: preview window flag
    args.preview = !(parser.cmd_option_exists("--no-preview") || parser.cmd_option_exists("-np"));

    // Cmd arg: output path
    args.output_path = parser.get_cmd_option("-o");
    if (args.output_path.empty()) {
        args.output_path = "output/" + date_time_string();
    }

    // Cmd arg: output format
    args.output_format = parser.get_cmd_option("-f");
    if (args.output_format.empty()) {
        args.output_format = "png";
    } else if (!any_of(args.output_format, "png", "jpg", "bmp", "tga", "hdr")) {
        error_with_message("Invalid output format: " + args.output_format + ". Supported formats are: png, jpg, bmp, tga, hdr");
    }

    // Cmd arg: render passes
    args.render_passes = parse_passes(parser.get_cmd_option("-p"));

    return args;
}

#endif