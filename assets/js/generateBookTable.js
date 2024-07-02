async function fetchMetadataAndGenerateTable() {
    try {
        const response = await fetch('/scripts/metadata.json');
        const metadata = await response.json();

        // Sort the metadata by completion_date in descending order
        metadata.sort((a, b) => {
            const dateA = a.completion_date ? new Date(a.completion_date) : new Date(0);
            const dateB = b.completion_date ? new Date(b.completion_date) : new Date(0);
            return dateB - dateA;
        });

        // Define the table and headers
        let table = '<table class="minimalist-table">';
        table += '<thead><tr><th>Title</th><th>Author</th><th>Recommend</th><th>Completed</th></tr></thead><tbody>';

        // Populate the table rows
        metadata.forEach(item => {
            const completionDate = item.completion_date ? new Date(item.completion_date) : null;
            const formattedCompletionDate = completionDate ? completionDate.toLocaleDateString('en-US', { year: 'numeric', month: 'short' }) : '';

            // Determine the recommend cell content
            let recommendCellContent = '';
            if (item.recommend === true) {
                recommendCellContent = '✅'; // Unicode character for check mark or any other emoji
            } else if (item.recommend === false) {
                recommendCellContent = '❌'; // bad mark
            } else if (item.recommend === 'neutral') {
                recommendCellContent = '🤷🏼‍♀️'
            } else {
                recommendCellContent = ''
            }

            table += `<tr>
                        <td>${item.title || ''}</td>
                        <td>${item.author || ''}</td>
                        <td>${recommendCellContent}</td>
                        <td>${formattedCompletionDate || ''}</td>
                      </tr>`;
        });

        table += '</tbody></table>';

        // Insert the table into the HTML element with ID 'metadata-table'
        document.getElementById('book-table').innerHTML = table;
    } catch (error) {
        console.error('Error fetching or parsing metadata:', error);
    }
}

// // Call the function to fetch metadata and generate the table
fetchMetadataAndGenerateTable();

