
    @staticmethod
    def _format_duration(duration_seconds):
        total_seconds = max(0, int(duration_seconds))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _update_presence_timers(self, now, active_track_ids, track_durations):
        stale_track_ids = []
        for tracker_id, presence in self.track_presence.items():
            if tracker_id in active_track_ids:
                continue

            if now - presence["last_seen"] >= self.track_timeout_seconds:
                self.completed_presence[tracker_id] = presence["last_seen"] - presence["first_seen"]
                stale_track_ids.append(tracker_id)

        for tracker_id in stale_track_ids:
            self.track_presence.pop(tracker_id, None)

        active_durations = list(track_durations.values())
        completed_durations = list(self.completed_presence.values())

        longest_active = max(active_durations, default=0.0)
        average_active = (
            sum(active_durations) / len(active_durations)
            if active_durations else 0.0
        )
        average_completed = (
            sum(completed_durations) / len(completed_durations)
            if completed_durations else 0.0
        )
        stall_count = sum(
            1 for duration in active_durations if duration >= self.stall_threshold_seconds
        )
        queue_count = sum(
            1 for duration in active_durations if duration >= self.queue_threshold_seconds
        )

        return {
            "active_count": len(active_durations),
            "stall_count": stall_count,
            "queue_count": queue_count,
            "longest_active": longest_active,
            "average_active": average_active,
            "average_completed": average_completed,
        }

    def export_path_data(self, file_path='data/customers_paths.csv'):
        """
        Export all tracked paths to a CSV database.
        """
        all_rows = []
        for track_id, path in self.tracked_paths.items():
            df_track = pd.DataFrame(path)
            df_track['track_id'] = track_id
            all_rows.append(df_track)
        
        if not all_rows:
            print("No path data to export.")
            return

        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return combined_df

    def generate_heatmap(self, output_file='data/heatmap.png'):
        """
        Create a density heatmap based on tracked positions.
        """
        # Initialize empty heatmap grid (same shape as input frame)
        # h, w = self.frame_shape
        heatmap_grid = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Accumulate density
        for track_id, path in self.tracked_paths.items():
            for point in path:
                # Round to nearest pixel (or use direct mapping if coordinates match)
                x = int(point['x'])
                y = int(point['y'])
                
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Increment pixel value
                    heatmap_grid[y, x] += 1

        # Normalize (optional)
        max_val = np.max(heatmap_grid)
        if max_val > 0:
            heatmap_grid = (heatmap_grid / max_val) * 255

        # Create visualization
        heat_map = cv2.applyColorMap(np.uint8(heatmap_grid), cv2.COLORMAP_JET)
        
        # Display result
        cv2.imwrite(output_file, heat_map)
        plt.figure(figsize=(15, 10))
        plt.imshow(heat_map, cmap='hot')
        plt.title(f"Customer Movement Heatmap (Max visits: {int(max_val)})")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(output_file.replace('.png', '.jpg')) # Save figure separately
        plt.close()
        
        print(f"Heatmap generated and saved to {output_file}")
        return heat_map

    def generate_lines(self, output_file='data/customer_paths.png'):
        """
        Draw customer paths with different colors on a single image.
        """
        # Initialize empty figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot each path with a unique color
        for track_id, path in self.tracked_paths.items():
            x_coords = [point['x'] for point in path]
            y_coords = [point['y'] for point in path]
            ax.plot(x_coords, y_coords, label=f'Track {track_id}', linewidth=2)
        
        # Set the aspect ratio to 'equal' to preserve the scale
        ax.set_aspect('equal')
        
        # Add labels and title
        plt.title("Customer Movement Paths")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        
        # Save the figure
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        print(f"Paths saved to {output_file}")     
        return fig